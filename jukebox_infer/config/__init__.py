"""Package-owned checkpoint metadata and cache inspection helpers."""
import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("checkpoints.toml")


def _expected_checkpoint_paths(model_name: str) -> list[str]:
    """Resolve `model_name` to the local checkpoint file paths it needs.

    Mirrors the exact resolution logic `make_models.load_checkpoint` /
    `make_models.download_checkpoints` use: each named hparam set's
    `restore_vqvae` / `restore_prior` remote URL maps to
    `~/.cache/<remote path with REMOTE_PREFIX stripped>`. Imported lazily so
    that importing `jukebox_infer.config` alone never pulls in torch.
    """
    from jukebox_infer.hparams import REMOTE_PREFIX, setup_hparams
    from jukebox_infer.make_models import MODELS

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    vqvae_name, *prior_names = MODELS[model_name]
    paths = []

    vqvae_hps = setup_hparams(vqvae_name, {})
    if vqvae_hps.restore_vqvae.startswith(REMOTE_PREFIX):
        remote_path = vqvae_hps.restore_vqvae
        paths.append(os.path.join(os.path.expanduser("~/.cache"), remote_path[len(REMOTE_PREFIX):]))

    for prior_name in prior_names:
        prior_hps = setup_hparams(prior_name, {})
        if prior_hps.restore_prior.startswith(REMOTE_PREFIX):
            remote_path = prior_hps.restore_prior
            paths.append(os.path.join(os.path.expanduser("~/.cache"), remote_path[len(REMOTE_PREFIX):]))

    return paths


def checkpoint_cache_info(model_name: str | None = None) -> dict:
    """Report whether `model_name`'s checkpoint files are actually cached.

    Checks the specific per-file paths the real download path reads/writes
    (via `utils.remote_utils.check_file_exists`, the same non-zero-size file
    check `make_models.load_checkpoint` uses to decide whether to download)
    -- not just whether the shared cache *directory* happens to exist. A
    directory can exist while a specific model's files are missing (or vice
    versa in a multi-model cache), so directory-existence alone cannot answer
    "is `model_name` cached".
    """
    from jukebox_infer.utils.remote_utils import check_file_exists

    path = Path.home() / ".cache" / "jukebox" / "models"
    if model_name is None:
        return {"directory": str(path), "exists": path.exists(), "model": model_name}

    expected_paths = _expected_checkpoint_paths(model_name)
    missing = [p for p in expected_paths if not check_file_exists(p)]
    return {
        "directory": str(path),
        "exists": path.exists(),
        "model": model_name,
        "files": expected_paths,
        "missing": missing,
        "cached": len(missing) == 0,
    }


def checkpoint_config_path() -> Path:
    return CONFIG_PATH

__all__ = ["CONFIG_PATH", "checkpoint_cache_info", "checkpoint_config_path"]
