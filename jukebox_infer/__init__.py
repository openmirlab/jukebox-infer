"""Jukebox-Infer: Minimal inference-only implementation of OpenAI Jukebox.

Public entry point: the `Jukebox` class (jukebox_infer.api) wraps model
loading, ancestral/primed sampling, and audio saving behind `.load()` /
`.generate()` / `.generate_from_audio()`. See README.md for the CLI and
Python API usage, and CLAUDE.md for the internal module layout.

Reads: jukebox_infer.api (Jukebox, set_seed), jukebox_infer.make_models
(download_checkpoints), jukebox_infer.__about__ (version)
"""

from importlib import import_module

from jukebox_infer.__about__ import __version__

__all__ = ["Jukebox", "JukeboxSession", "download_checkpoints", "set_seed", "__version__"]

# Keep the package door light: importing ``jukebox_infer.hparams`` or a
# SheetSage integration must not construct the full API/audio stack.  The
# legacy root exports remain available through PEP 562 lazy attribute access.
_LAZY_EXPORTS = {
    "Jukebox": ("jukebox_infer.api", "Jukebox"),
    "JukeboxSession": ("jukebox_infer.api", "JukeboxSession"),
    "set_seed": ("jukebox_infer.api", "set_seed"),
    "download_checkpoints": ("jukebox_infer.make_models", "download_checkpoints"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
