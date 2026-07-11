"""Jukebox-Infer: Minimal inference-only implementation of OpenAI Jukebox.

Public entry point: the `Jukebox` class (jukebox_infer.api) wraps model
loading, ancestral/primed sampling, and audio saving behind `.load()` /
`.generate()` / `.generate_from_audio()`. See README.md for the CLI and
Python API usage, and CLAUDE.md for the internal module layout.

Reads: jukebox_infer.api (Jukebox, set_seed), jukebox_infer.make_models
(download_checkpoints), jukebox_infer.__about__ (version)
"""

from jukebox_infer.__about__ import __version__
from jukebox_infer.api import Jukebox, set_seed
from jukebox_infer.make_models import download_checkpoints

__all__ = ["Jukebox", "download_checkpoints", "set_seed", "__version__"]
