"""Load config.yaml (override the location with the MTG_CONFIG environment variable)."""
import os
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path: str | None = None) -> dict:
    config_path = Path(path or os.environ.get("MTG_CONFIG", DEFAULT_CONFIG_PATH))
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


def project_path(relative: str) -> Path:
    """Resolve a path from the config against the project root (absolute paths are kept)."""
    return PROJECT_ROOT / relative


CONFIG = load_config()

