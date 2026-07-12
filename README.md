# Jukebox-Infer

**Inference-only implementation of OpenAI Jukebox for modern PyTorch (2.13+)**

[![PyPI](https://img.shields.io/pypi/v/jukebox-infer)](https://pypi.org/project/jukebox-infer/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-quality music generation models for creating music from scratch or continuing existing audio tracks.

---

## 📌 Overview

**Jukebox-Infer** is a streamlined, inference-only version of [OpenAI Jukebox](https://github.com/openai/jukebox), optimized for PyTorch 2.13+ with minimal dependencies.

> **Note**: This project is based on [OpenAI Jukebox](https://github.com/openai/jukebox). All credit for the original model and research belongs to OpenAI and the Jukebox authors.

---

## 🎉 What's New

- **v0.1.1** (Latest): Fixed the installed `jukebox-infer` console script, which previously failed on every install with "Error: quick_infer.py not found"; removed a dead `apex` import remnant from the transformer code
- **v0.1.0**: Initial release - Clean inference-only implementation extracted from OpenAI Jukebox

---

## ✨ Features

- ✅ **100% Parity Verified** - VQ-VAE features identical to original Jukebox (see [Parity Verification](docs/PARITY_VERIFICATION.md))
- ✅ **Inference-only** - No training code, significantly reduced codebase (~47% reduction)
- ✅ **Modern PyTorch** - Compatible with PyTorch 2.13+
- ✅ **Single-GPU** - No MPI or distributed dependencies
- ✅ **Minimal dependencies** - Removed tensorboardX, apex, and training-specific libs
- ✅ **Auto-download** - Automatic checkpoint downloads on first use
- ✅ **GPU acceleration** - Full CUDA support with optimized device management
- ✅ **Simple API** - High-level `Jukebox` class for easy music generation
- ✅ **Audio continuation** - Support for primed sampling from audio prompts

---

---

## 🚀 Quick Start

### Installation

**From PyPI:**

```bash
# Using pip
pip install jukebox-infer

# Using uv (recommended - faster)
uv pip install jukebox-infer

# Or add to your project with uv
uv add jukebox-infer
```

**For Development:**

```bash
# Clone the repository
git clone https://github.com/openmirlab/jukebox-infer.git
cd jukebox-infer

# Install in editable mode
pip install -e .

# Or with uv
uv pip install -e .
```

> **Package:** https://pypi.org/project/jukebox-infer/
>
> **Note:** If you're setting up both the original Jukebox and jukebox-infer for comparison testing, see [../JUKEBOX_SETUP.md](../JUKEBOX_SETUP.md) for detailed environment setup instructions.

### Command-Line Interface (Fastest)

If you installed via pip/uv, a `jukebox-infer` console script is available
directly on your PATH:

```bash
# Basic generation (default: 20 seconds, The Beatles, Rock)
jukebox-infer

# Custom artist and genre
jukebox-infer --artist "Taylor Swift" --genre "Pop" --duration 30

# Audio continuation from existing audio
jukebox-infer --prompt input.wav --prompt-duration 5 --duration 20 --output continuation.wav

# See all options
jukebox-infer --help
```

If you're working from a repo checkout instead, `quick_infer.py` is an
equivalent standalone entry point (both call the same code in
`jukebox_infer/quick_infer.py`):

```bash
python quick_infer.py --artist "Taylor Swift" --genre "Pop" --duration 30
python quick_infer.py --help
```

### Simple API (Recommended for Python)

```python
from jukebox_infer import Jukebox

# Initialize model (checkpoints auto-download on first use)
model = Jukebox(model_name="1b_lyrics", device="cuda")
model.load(sample_length_in_seconds=20)

# Generate music
audio = model.generate(
    artist="The Beatles",
    genre="Rock",
    duration_seconds=20,
    output_path="output.wav"
)
```

### Audio Continuation

**CLI:**
```bash
python quick_infer.py --prompt input.wav --prompt-duration 5 --duration 20 --output continuation.wav
```

**Python API:**
```python
from jukebox_infer import Jukebox

model = Jukebox(model_name="1b_lyrics", device="cuda")
model.load(sample_length_in_seconds=20)

# Continue from existing audio
audio = model.generate_from_audio(
    prompt_audio="input.wav",
    prompt_duration=5,  # Use first 5 seconds as prompt
    total_duration=20,  # Generate 20 seconds total
    output_path="continuation.wav"
)
```

---

---

## 📦 Download Checkpoints

Checkpoints are **automatically downloaded** when you first use a model. No manual download needed!

If you prefer to pre-download checkpoints manually:

```bash
# Option 1: Use the download script
bash download_checkpoints.sh

# Option 2: Use Python API
from jukebox_infer import download_checkpoints
download_checkpoints('1b_lyrics')  # Downloads ~6.2GB
```

Checkpoints are cached in `~/.cache/jukebox/models/`:
- VQ-VAE (7.4MB) - shared encoder/decoder
- Prior level 0 & 1 (4.4GB) - shared upsamplers  
- Prior level 2 (1.8GB) - 1b_lyrics top-level model

---

## 🎵 Available Models

| Model | Parameters | Download Size | VRAM | Description |
|-------|-----------|---------------|------|-------------|
| **`1b_lyrics`** | 1B | ~6.2GB | ~12GB | Lyrics conditioning support |

---

## 📋 Requirements

- **Python**: ≥3.10
- **PyTorch**: ≥2.13.0
- **GPU**: CUDA-capable GPU (16GB+ VRAM recommended for 1b_lyrics)
- **OS**: Linux, macOS, Windows

---

---

## ⚡ Performance

Generation is intentionally slow due to autoregressive nature:
- **~5-15 seconds per second of audio** on RTX 4090 (with GPU acceleration)
- **18 seconds**: ~3-5 minutes
- **60 seconds**: ~5-15 minutes

This matches the original implementation's performance characteristics.

> **Note**: Generation speed depends on GPU, model size, and generation length. The autoregressive nature means longer generations take proportionally longer.

---

## 📚 Documentation

- **[PARITY_VERIFICATION.md](docs/PARITY_VERIFICATION.md)** - ✅ **100% parity verification** with original Jukebox
- **[CHECKPOINT_ARCHITECTURE.md](docs/CHECKPOINT_ARCHITECTURE.md)** - Details on checkpoint structure and sharing between models
- **[Development Guidelines](docs/dev/PRINCIPLES.md)** - Development principles, code style, and contribution guidelines

---

## 🏗️ Project Structure

```
jukebox-infer/
├── jukebox_infer/      # Main package
│   ├── api.py         # High-level Jukebox API
│   ├── cli.py         # Console-script entry point (thin re-export)
│   ├── quick_infer.py # CLI argument parsing + generation drive loop
│   ├── make_models.py # Model loading and checkpoint management
│   ├── sample.py      # Sampling functions
│   ├── prior/         # Prior model implementations
│   ├── vqvae/         # VQ-VAE encoder/decoder
│   ├── transformer/   # Transformer architecture
│   └── data/          # Data processing utilities + checkpoints.json (provenance)
├── docs/              # Documentation
│   ├── PARITY_VERIFICATION.md      # ✅ 100% parity proof
│   └── CHECKPOINT_ARCHITECTURE.md
├── tests/             # Import/CLI/hparams smoke tests + network-marked liveness test
├── tools/             # check_weights_liveness.py
├── examples/          # Example scripts
├── quick_infer.py     # Standalone shim -> jukebox_infer.quick_infer.main
├── download_checkpoints.sh  # Manual download script
├── pyproject.toml
├── CLAUDE.md
├── LICENSE
└── README.md
```

---

---

## ✅ Parity Verification

**jukebox-infer has been rigorously verified to produce 100% identical VQ-VAE features compared to the original OpenAI Jukebox.**

### Test Results

| Metric | Result |
|--------|--------|
| **max \|Δ\|** | 0.000000e+00 |
| **mean \|Δ\|** | 0.000000e+00 |
| **Feature shape** | (1, 6146) - identical |
| **Feature range** | [8, 2035] - identical |
| **Parity status** | ✅ **100% VERIFIED** |

### What This Means

- ✅ **Perfect numerical match** - Zero difference in VQ-VAE feature extraction
- ✅ **Drop-in replacement** - Can completely replace original Jukebox for feature extraction
- ✅ **No accuracy loss** - Maintains 100% fidelity to original implementation
- ✅ **Research confidence** - Validated for academic and production use

### Testing Methodology

Parity was verified using:
- Multiple audio durations (5s, 20s)
- Identical official OpenAI checkpoints
- Rigorous numerical comparison (rtol=1e-4, atol=1e-6)
- Both CPU and GPU modes tested

**For full details, see [PARITY_VERIFICATION.md](docs/PARITY_VERIFICATION.md)**

---

## 🙏 Acknowledgments

### Original Research by OpenAI

**Jukebox-Infer** is built upon the groundbreaking work of [OpenAI Jukebox](https://github.com/openai/jukebox). The original Jukebox represents a major advancement in music generation, achieving state-of-the-art results through innovative hierarchical VQ-VAE and transformer architectures.

### Research Paper

**[Jukebox: A Generative Model for Music](https://arxiv.org/abs/2005.00341)**

This seminal work introduced hierarchical music generation with conditioning on artist, genre, and lyrics, enabling high-quality music generation at multiple time scales.

### Original Authors

- Prafulla Dhariwal
- Heewoo Jun
- Christine Payne
- Jong Wook Kim
- Alec Radford
- Ilya Sutskever

### About This Implementation

> **Note**: The original Jukebox repository is no longer actively maintained. This package was created to continue the excellent work by providing ongoing maintenance and PyTorch 2.13+ compatibility for the inference capabilities, while preserving 100% of the original model quality and algorithms.

**What we maintain:**
- PyTorch 2.13+ compatibility
- Modern dependency management
- Inference-only packaging
- GPU optimization

**What remains unchanged:**
- All model architectures (100% original)
- All generation algorithms (100% original)
- All model weights (100% original)
- VQ-VAE feature extraction (✅ **100% parity verified** - see [PARITY_VERIFICATION.md](docs/PARITY_VERIFICATION.md))

---

## 📄 Citation

Please cite using the following bibtex entry:

```bibtex
@article{dhariwal2020jukebox,
  title={Jukebox: A Generative Model for Music},
  author={Dhariwal, Prafulla and Jun, Heewoo and Payne, Christine and Kim, Jong Wook and Radford, Alec and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2005.00341},
  year={2020}
}
```

**If you use Jukebox-Infer in your research, please cite the original Jukebox paper above.** This package is merely a maintenance fork to ensure continued compatibility with modern PyTorch versions - all credit for the models, algorithms, and research belongs to the original authors.

---

## 📄 License

**MIT License** (same as original Jukebox)

Copyright (c) 2020 OpenAI (Original Jukebox)
Copyright (c) 2025 (Jukebox-Infer modifications)

See [LICENSE](LICENSE) for details.

This project includes code adapted from [OpenAI Jukebox](https://github.com/openai/jukebox) (MIT License, Copyright 2020 OpenAI).

---

## ⚠️ Limitations

- **Inference only** - No training capabilities
- **Single GPU** - No distributed inference
- **Slow generation** - Autoregressive model, ~5-15 seconds per second of audio
- **Minimum duration** - 1b_lyrics requires 17.84-600 seconds
- **Large checkpoints** - ~6.2GB download required

---

## 🤝 Contributing

We welcome contributions! Please:

1. Read [docs/dev/PRINCIPLES.md](docs/dev/PRINCIPLES.md) for development guidelines
2. Follow the code style (ruff/black)
3. Add tests for new features
4. Update documentation
5. Submit PRs with clear descriptions

### Development Setup

```bash
# Install dependencies with UV
uv sync

# Run quick inference script
uv run python quick_infer.py

# Format and lint code
uv run ruff format . && uv run ruff check .
```

See [docs/dev/PRINCIPLES.md](docs/dev/PRINCIPLES.md) for detailed development guidelines.

---

## 📞 Support

For issues and questions:
- **GitHub Issues**: [github.com/openmirlab/jukebox-infer/issues](https://github.com/openmirlab/jukebox-infer/issues)
- **Documentation**: `docs/`
- **Examples**: `examples/`

---

**Made with ❤️ for the ML community**

Based on the excellent work by OpenAI and the Jukebox authors.
