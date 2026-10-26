from datetime import datetime

import numpy as np
import torch

from src.cnn import CellNowcastNet, CycleData, Sample, block_sum, list_samples, make_input


def make_fake_cycle(ir_value=250.0, wv_value=240.0, flash_value=0.0, shape=(10, 10)):
    return CycleData(
        ir=np.full(shape, ir_value, dtype="float32"),
        wv=np.full(shape, wv_value, dtype="float32"),
        flash=np.full(shape, flash_value, dtype="float32"),
        cell_flash_sum=np.zeros((2, 2)),
    )


def test_block_sum_drops_trailing_rows_and_columns():
    field = np.ones((12, 13))
    cells = block_sum(field, 5)
    assert cells.shape == (2, 2)
    assert (cells == 25).all()


def test_network_outputs_one_logit_per_cell():
    network = CellNowcastNet(input_channels=4, cell_size=5)
    logits = network(torch.zeros(2, 4, 52, 58))
    assert tuple(logits.shape) == (2, 10, 11)   # 52 // 5 and 58 // 5


def test_list_samples_needs_past_and_target_cycles():
    cycles = {datetime(2026, 7, 20, h, m): make_fake_cycle() for h, m in [(12, 0), (12, 30), (13, 0)]}
    samples = list_samples(cycles, lead_cycles=3, past_cycles=3)
    assert samples == [Sample(datetime(2026, 7, 20, 12, 30), datetime(2026, 7, 20, 12, 0), datetime(2026, 7, 20, 13, 0))]


def test_make_input_channels_are_cooling_and_water_vapour_difference():
    past_time, image_time = datetime(2026, 7, 20, 12, 0), datetime(2026, 7, 20, 12, 30)
    cycles = {
        past_time: make_fake_cycle(ir_value=260.0),
        image_time: make_fake_cycle(ir_value=250.0, wv_value=240.0, flash_value=0.5),
    }
    sample = Sample(image_time, past_time, image_time)

    without_lightning = make_input(cycles, sample, use_lightning=False)
    with_lightning = make_input(cycles, sample, use_lightning=True)

    assert without_lightning.shape == (4, 10, 10)
    assert with_lightning.shape == (6, 10, 10)
    assert np.allclose(without_lightning[2], -10.0)   # 250 K now vs 260 K before
    assert np.allclose(without_lightning[3], -10.0)   # 240 K water vapour minus 250 K infrared
    assert np.allclose(with_lightning[4], np.log1p(0.5))