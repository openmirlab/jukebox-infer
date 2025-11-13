"""
Jukebox-Infer: Minimal inference-only implementation of OpenAI Jukebox

This package provides a streamlined version of Jukebox for music generation,
optimized for PyTorch 2.7+ and single-GPU inference.
"""

__version__ = "0.1.0"

from jukebox_infer.api import Jukebox
from jukebox_infer.make_models import download_checkpoints

__all__ = ["Jukebox", "download_checkpoints", "__version__"]
