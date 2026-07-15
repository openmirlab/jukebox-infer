"""Package-owned checkpoint metadata and cache inspection helpers."""
from pathlib import Path
import json

CONFIG_PATH = Path(__file__).with_name("checkpoints.toml")

def checkpoint_cache_info(model_name: str | None = None) -> dict:
    path = Path.home() / ".cache" / "jukebox" / "models"
    return {"directory": str(path), "exists": path.exists(), "model": model_name}

def checkpoint_config_path() -> Path:
    return CONFIG_PATH

__all__ = ["CONFIG_PATH", "checkpoint_cache_info", "checkpoint_config_path"]
