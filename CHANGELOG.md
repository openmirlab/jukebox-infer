# Changelog

All notable changes to Jukebox-Infer will be documented in this file.

## [0.1.0] - 2025-01-XX

### Initial Release

#### Added
- **Inference-only implementation** - Extracted from OpenAI Jukebox, removing all training code
- **Modern PyTorch 2.7+ support** - Compatible with latest PyTorch versions
- **High-level API** - Simple `Jukebox` class for easy music generation
- **Automatic checkpoint download** - Checkpoints download automatically on first use
- **Manual download options** - Shell script and Python API for pre-downloading checkpoints
- **GPU acceleration** - Full CUDA support with proper device management
- **Audio continuation** - Support for primed sampling from audio prompts

#### Features
- VQ-VAE encoder/decoder for audio tokenization
- Multi-level prior models (3 levels) for hierarchical generation
- Ancestral and primed sampling modes
- Artist and genre conditioning
- Lyrics support (for lyrics-capable models)

#### Models
- `1b_lyrics` - 1 billion parameter model with lyrics conditioning (~6.2GB checkpoints)

#### Technical Improvements
- Removed training dependencies (MPI, distributed training, tensorboardX, apex)
- Single-GPU inference optimized
- Fixed device placement bugs (all models now correctly use GPU)
- Optimized batch sizes for better GPU utilization
- Clean separation of concerns (API, sampling, model loading)

#### Documentation
- Comprehensive README with quick start guide
- Checkpoint architecture documentation
- Example scripts for basic generation and audio continuation

#### Credits
- Based on OpenAI Jukebox (https://github.com/openai/jukebox)
- Original paper: "Jukebox: A Generative Model for Music" (https://arxiv.org/abs/2005.00341)
