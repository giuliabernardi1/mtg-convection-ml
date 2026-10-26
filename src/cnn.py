"""Small fully convolutional nowcasting network, evaluated with leave-one-block-out.

Inputs: native 2 km fields at time t and t - past (IR 10.5, WV 6.3, optionally LI accumulation).
Output: one logit per coarse cell (cell_size x cell_size native pixels) for lightning at t + lead,
with the same cells and labels as the tabular baseline (src.labels).
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.metrics import average_precision_score
from torch import nn

from src.config import CONFIG
from src.evaluation import categorical_scores
from src.labels import ACCUMULATION_THRESHOLD, CELL_SIZE, CYCLE_DURATION, index_cycles
from src.preprocessing import AREA_EXTENT
from src.train import PROBABILITY_THRESHOLDS, assign_blocks

log = logging.getLogger(__name__)
MAX_POSITIVE_WEIGHT = CONFIG["cnn"]["max_positive_weight"]


@dataclass
class CycleData:
    ir: np.ndarray               # native pixels, K
    wv: np.ndarray
    flash: np.ndarray            # native accumulation, missing = 0
    cell_flash_sum: np.ndarray   # accumulation summed over cells


@dataclass(frozen=True)
class Sample:
    image_time: datetime
    past_time: datetime
    target_time: datetime


def block_sum(field: np.ndarray, cell_size: int) -> np.ndarray:
    """Sum non-overlapping cells; trailing rows/columns are dropped (like coarsen(boundary='trim'))."""
    n_rows = field.shape[0] // cell_size * cell_size
    n_cols = field.shape[1] // cell_size * cell_size
    cells = field[:n_rows, :n_cols].reshape(n_rows // cell_size, cell_size, n_cols // cell_size, cell_size)
    return cells.sum(axis=(1, 3))


def load_cycles(processed_dir: Path, cell_size: int):
    """All processed cycles in memory, plus the boolean mask of cells inside the region of interest."""
    cycles = {}
    region_mask = None
    for cycle_time, path in index_cycles(processed_dir).items():
        with xr.open_dataset(path) as dataset:
            flash = np.nan_to_num(dataset["flash_accumulation"].values, nan=0.0).astype("float32")
            cycles[cycle_time] = CycleData(
                ir=dataset["ir_105"].values.astype("float32"),
                wv=dataset["wv_63"].values.astype("float32"),
                flash=flash,
                cell_flash_sum=block_sum(flash, cell_size),
            )
            if region_mask is None:  # the pixel window is the same for every cycle
                cell_longitudes = block_sum(dataset["longitude"].values, cell_size) / cell_size**2
                cell_latitudes = block_sum(dataset["latitude"].values, cell_size) / cell_size**2
                lon_min, lat_min, lon_max, lat_max = AREA_EXTENT
                region_mask = (
                    (cell_longitudes >= lon_min) & (cell_longitudes <= lon_max)
                    & (cell_latitudes >= lat_min) & (cell_latitudes <= lat_max)
                )
    return cycles, region_mask


def list_samples(cycles, lead_cycles: int, past_cycles: int) -> list[Sample]:
    """Same pairing rule as the tabular dataset: needs the past cycle and the target cycle."""
    samples = []
    for image_time in sorted(cycles):
        target_time = image_time + lead_cycles * CYCLE_DURATION
        past_time = image_time - past_cycles * CYCLE_DURATION
        if target_time in cycles and past_time in cycles:
            samples.append(Sample(image_time, past_time, target_time))
    return samples


def make_input(cycles, sample: Sample, use_lightning: bool) -> np.ndarray:
    now, past = cycles[sample.image_time], cycles[sample.past_time]
    channels = [now.ir, now.wv, now.ir - past.ir, now.wv - now.ir]
    if use_lightning:
        channels += [np.log1p(now.flash), np.log1p(past.flash)]
    return np.stack(channels).astype("float32")


def compute_channel_stats(cycles, samples, use_lightning: bool, max_samples: int = 60):
    chosen_indices = np.random.default_rng(0).choice(
        len(samples), size=min(max_samples, len(samples)), replace=False
    )
    stacked_inputs = np.stack([make_input(cycles, samples[i], use_lightning) for i in chosen_indices])
    channel_mean = np.nanmean(stacked_inputs, axis=(0, 2, 3)).astype("float32")
    channel_std = (np.nanstd(stacked_inputs, axis=(0, 2, 3)) + 1e-6).astype("float32")
    return channel_mean, channel_std


def make_inputs(cycles, samples, use_lightning, channel_mean, channel_std) -> torch.Tensor:
    inputs = np.stack([make_input(cycles, sample, use_lightning) for sample in samples])
    normalised_inputs = (inputs - channel_mean[None, :, None, None]) / channel_std[None, :, None, None]
    return torch.from_numpy(np.nan_to_num(normalised_inputs, nan=0.0).astype("float32"))


def targets_for(cycles, samples, threshold: float) -> np.ndarray:
    return np.stack([cycles[sample.target_time].cell_flash_sum >= threshold for sample in samples])


class CellNowcastNet(nn.Module):
    """Native-resolution convolutions, max-pool to cells, then dilated convolutions for spatial context."""

    def __init__(self, input_channels: int, cell_size: int):
        super().__init__()
        self.to_cells = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(cell_size),  # floor: drops trailing rows/columns like block_sum
        )
        self.context = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.context(self.to_cells(inputs)).squeeze(1)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_network(cycles, train_samples, region_mask, use_lightning, threshold, cell_size,
                  epochs, device, batch_size=16, learning_rate=1e-3):
    torch.manual_seed(0)
    channel_mean, channel_std = compute_channel_stats(cycles, train_samples, use_lightning)
    train_targets = targets_for(cycles, train_samples, threshold)
    positive_count = int(train_targets[:, region_mask].sum())
    negative_count = train_targets[:, region_mask].size - positive_count
    positive_weight = min(MAX_POSITIVE_WEIGHT, negative_count / max(positive_count, 1))

    model = CellNowcastNet(len(channel_mean), cell_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(positive_weight, device=device))
    region_weights = torch.from_numpy(region_mask.astype("float32")).to(device)
    shuffle_generator = np.random.default_rng(0)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        shuffled_indices = shuffle_generator.permutation(len(train_samples))
        for start in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[start:start + batch_size]
            batch_samples = [train_samples[i] for i in batch_indices]
            inputs = make_inputs(cycles, batch_samples, use_lightning, channel_mean, channel_std).to(device)
            targets = torch.from_numpy(train_targets[batch_indices].astype("float32")).to(device)
            losses = loss_function(model(inputs), targets) * region_weights
            loss = losses.sum() / (region_weights.sum() * len(batch_samples))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * len(batch_samples)
        log.info("epoch %d/%d | loss %.5f", epoch + 1, epochs, epoch_loss / len(train_samples))
    return model, channel_mean, channel_std


@torch.no_grad()
def predict_probabilities(model, cycles, samples, use_lightning, channel_mean, channel_std, device,
                          batch_size=16) -> np.ndarray:
    model.eval()
    probability_batches = []
    for start in range(0, len(samples), batch_size):
        inputs = make_inputs(cycles, samples[start:start + batch_size], use_lightning, channel_mean, channel_std)
        probability_batches.append(torch.sigmoid(model(inputs.to(device))).cpu().numpy())
    return np.concatenate(probability_batches)


def choose_probability_threshold(probabilities, targets, region_mask) -> float:
    csi_per_threshold = [
        categorical_scores(targets[:, region_mask], probabilities[:, region_mask] >= threshold)["csi"]
        for threshold in PROBABILITY_THRESHOLDS
    ]
    if np.all(np.isnan(csi_per_threshold)):
        return 0.5
    return float(PROBABILITY_THRESHOLDS[int(np.nanargmax(csi_per_threshold))])


def run_leave_one_block_out(processed_dir: Path, use_lightning: bool, lead_cycles: int = 3,
                            past_cycles: int = 3, cell_size: int = CELL_SIZE,
                            threshold: float = ACCUMULATION_THRESHOLD, max_gap_days: int = 2,
                            epochs: int = 12, subsample_step: int = 1) -> pd.DataFrame:
    """For each test block: train on the other blocks except the smallest one, which is held out
    to choose the probability threshold. subsample_step > 1 trains on every k-th sample (smoke runs)."""
    device = select_device()
    cycles, region_mask = load_cycles(processed_dir, cell_size)
    samples = list_samples(cycles, lead_cycles, past_cycles)
    days = pd.Series([sample.image_time.date().isoformat() for sample in samples])
    block_ids = assign_blocks(days, max_gap_days).to_numpy()
    unique_blocks = sorted(set(block_ids))
    if len(unique_blocks) < 3:
        raise ValueError(f"Need at least 3 blocks (train, validation, test), found {len(unique_blocks)}")
    log.info("device %s | %d samples | blocks: %s", device, len(samples),
             {int(b): sorted(set(days[block_ids == b])) for b in unique_blocks})

    fold_results = []
    for test_block in unique_blocks:
        other_blocks = [b for b in unique_blocks if b != test_block]
        validation_block = min(other_blocks, key=lambda b: int((block_ids == b).sum()))
        pick = lambda wanted: [s for s, b in zip(samples, block_ids) if wanted(b)]
        test_samples = pick(lambda b: b == test_block)
        validation_samples = pick(lambda b: b == validation_block)
        train_samples = pick(lambda b: b not in (test_block, validation_block))[::subsample_step]

        model, channel_mean, channel_std = train_network(
            cycles, train_samples, region_mask, use_lightning, threshold, cell_size, epochs, device
        )
        validation_probabilities = predict_probabilities(
            model, cycles, validation_samples, use_lightning, channel_mean, channel_std, device)
        probability_threshold = choose_probability_threshold(
            validation_probabilities, targets_for(cycles, validation_samples, threshold), region_mask)

        test_probabilities = predict_probabilities(
            model, cycles, test_samples, use_lightning, channel_mean, channel_std, device)
        test_targets = targets_for(cycles, test_samples, threshold)
        test_persistence = np.stack([cycles[s.image_time].cell_flash_sum >= threshold for s in test_samples])

        observed_cells = test_targets[:, region_mask]
        model_scores = categorical_scores(observed_cells, test_probabilities[:, region_mask] >= probability_threshold)
        persistence_scores = categorical_scores(observed_cells, test_persistence[:, region_mask])
        test_days = sorted({s.image_time.date().isoformat() for s in test_samples})
        fold_results.append({
            "test_block": int(test_block),
            "test_days": f"{test_days[0]}..{test_days[-1]} ({len(test_days)} days)",
            "threshold": probability_threshold,
            "pr_auc": average_precision_score(observed_cells.ravel(), test_probabilities[:, region_mask].ravel()),
            **{f"model_{name}": value for name, value in model_scores.items()},
            **{f"persistence_{name}": value for name, value in persistence_scores.items()},
        })
        log.info("block %s | CSI model %.3f vs persistence %.3f", test_block,
                 model_scores["csi"], persistence_scores["csi"])
    return pd.DataFrame(fold_results)