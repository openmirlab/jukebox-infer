"""Runtime checkpoint catalog and cache resolver (★ load-bearing).

The packaged TOML file is the sole production owner of official checkpoint
URLs and their cache paths. The legacy JSON registry is source-only parity
data for liveness tooling and is never read by runtime code.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

CONFIG_PATH = Path(__file__).with_name("checkpoints.toml")


def checkpoint_catalog() -> dict[str, dict[str, Any]]:
    """Load the packaged catalog and validate its stable runtime keys."""
    with CONFIG_PATH.open("rb") as handle:
        models = tomllib.load(handle).get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"invalid checkpoint catalog: {CONFIG_PATH}")
    for name, record in models.items():
        if not isinstance(record, dict) or not record.get("url"):
            raise ValueError(f"invalid checkpoint record: {name}")
        if record.get("integrity") != "unavailable":
            raise ValueError(f"unsupported integrity record: {name}")
    return models


def checkpoint_url(name: str) -> str:
    """Return one official runtime checkpoint URL by stable catalog key."""
    try:
        return str(checkpoint_catalog()[name]["url"])
    except KeyError as error:
        raise ValueError(f"unknown checkpoint: {name}") from error


def resolve_checkpoint_path(remote_url: str) -> str:
    """Map a catalog URL to the exact cache path used by downloading/loading."""
    from jukebox_infer.hparams import REMOTE_PREFIX

    if remote_url not in {record["url"] for record in checkpoint_catalog().values()}:
        raise ValueError(f"URL is not in the runtime checkpoint catalog: {remote_url}")
    if not remote_url.startswith(REMOTE_PREFIX):
        raise ValueError(f"unsupported checkpoint URL: {remote_url}")
    return os.path.join(os.path.expanduser("~/.cache"), remote_url[len(REMOTE_PREFIX):])


def _expected_checkpoint_paths(model_name: str) -> list[str]:
    """Resolve a model through its TOML-backed hparam checkpoint URLs."""
    from jukebox_infer.hparams import setup_hparams
    from jukebox_infer.make_models import MODELS

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    vqvae_name, *prior_names = MODELS[model_name]
    hparams = [setup_hparams(vqvae_name, {})] + [setup_hparams(name, {}) for name in prior_names]
    return [resolve_checkpoint_path(hps.get("restore_vqvae") or hps.get("restore_prior")) for hps in hparams]


def checkpoint_cache_info(model_name: str | None = None) -> dict[str, Any]:
    """Report cache state via the same TOML resolver model loading uses."""
    from jukebox_infer.utils.remote_utils import check_file_exists

    directory = Path.home() / ".cache" / "jukebox" / "models"
    if model_name is None:
        return {"directory": str(directory), "exists": directory.exists(), "model": None}
    files = _expected_checkpoint_paths(model_name)
    missing = [path for path in files if not check_file_exists(path)]
    return {"directory": str(directory), "exists": directory.exists(), "model": model_name,
            "files": files, "missing": missing, "cached": not missing}


def checkpoint_config_path() -> Path:
    return CONFIG_PATH


__all__ = ["CONFIG_PATH", "checkpoint_cache_info", "checkpoint_catalog", "checkpoint_config_path", "checkpoint_url", "resolve_checkpoint_path"]
