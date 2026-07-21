# Jukebox-Infer

**Inference-only implementation of OpenAI Jukebox for modern PyTorch (2.13+)**

[![PyPI](https://img.shields.io/pypi/v/jukebox-infer)](https://pypi.org/project/jukebox-infer/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-quality music generation models for creating music from scratch or continuing existing audio tracks.

---

## Why this exists

[OpenAI Jukebox](https://github.com/openai/jukebox) is a landmark hierarchical
VQ-VAE + transformer music generation model, but the upstream repository is
no longer actively maintained: it targets an old PyTorch release, depends on
MPI and `apex` for distributed training that inference users don't need, and
has not picked up compatibility fixes for modern PyTorch.

**Jukebox-Infer** re-provides just the inference path — checkpoint download,
model construction, and ancestral/primed sampling — as a small,
single-GPU-only package that installs cleanly on PyTorch 2.13+ with no MPI or
`apex` dependency (~47% smaller codebase than upstream). All model
architectures, generation algorithms, and weights are unchanged from the
original; see [Parity Verification](docs/PARITY_VERIFICATION.md) for a
rigorous numerical proof that the VQ-VAE feature extraction is bit-identical
to upstream.

For the full research codebase (training, MPI, `apex`), use
[openai/jukebox](https://github.com/openai/jukebox) directly.

---

## Acknowledgments

**Jukebox-Infer** is built entirely on the model, algorithms, and weights
released by the original Jukebox project. This package's own contribution is
limited to the maintenance work described in [Why this exists](#why-this-exists)
above — packaging, dependency modernization, and inference-only trimming.

- **[openai/jukebox](https://github.com/openai/jukebox)** — the original
  research codebase this package is derived from.
- **Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec
  Radford, Ilya Sutskever** — original authors of the
  [Jukebox paper](https://arxiv.org/abs/2005.00341) and model (see
  [Citation](#citation) below for the full bibtex).
- **Checkpoint host**: original weights are served from OpenAI's own
  `https://openaipublic.azureedge.net/` CDN (auto-downloaded by this
  package — see [What this project will NEVER bundle](#what-this-project-will-never-bundle)).

All credit for the model architecture, generation algorithms, and released
weights belongs to OpenAI and the original Jukebox authors.

## Citation

If you use Jukebox-Infer in your research, please cite the original Jukebox
paper — this package is a maintenance fork, not new research:

```bibtex
@article{dhariwal2020jukebox,
  title={Jukebox: A Generative Model for Music},
  author={Dhariwal, Prafulla and Jun, Heewoo and Payne, Christine and Kim, Jong Wook and Radford, Alec and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2005.00341},
  year={2020}
}
```

---

## Features

- **100% Parity Verified** - VQ-VAE features identical to original Jukebox (see [Parity Verification](docs/PARITY_VERIFICATION.md))
- **Inference-only** - No training code, significantly reduced codebase (~47% reduction)
- **Modern PyTorch** - Compatible with PyTorch 2.13+
- **Single-GPU** - No MPI or distributed dependencies
- **Minimal dependencies** - Removed tensorboardX, apex, and training-specific libs
- **Auto-download** - Automatic checkpoint downloads on first use
- **GPU acceleration** - Full CUDA support with optimized device management
- **Simple API** - High-level `Jukebox` class for easy music generation
- **Audio continuation** - Support for primed sampling from audio prompts

---

## Scope

**In scope, and exercised by every documented workflow in this README:**

- The `1b_lyrics` model (~6.2GB checkpoints) — ancestral sampling and
  audio-primed continuation, artist/genre/lyrics conditioning.
- Single-GPU CUDA inference (16GB+ VRAM recommended), CPU fallback.

**Out of scope, forever:**

- **Training** — no training loop, no data pipeline; this is an
  inference-only package by design.
- **Distributed/multi-GPU inference** — no MPI, no `apex`; all upstream
  distributed-training machinery was removed.

**Present in code but untested/unsupported — do not rely on these:**

- The `5b` and `5b_lyrics` model entries exist in `hparams.py`'s registry
  (inherited from upstream) but are **not exercised by any documented
  workflow, test, or CLI default in this package**. They may or may not
  work; treat them as unmaintained until someone verifies and documents
  them.

---

## Install

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

---

## Quick Start

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

## Available Models

| Model | Parameters | Download Size | VRAM | Description |
|-------|-----------|---------------|------|-------------|
| **`1b_lyrics`** | 1B | ~6.2GB | ~12GB | Lyrics conditioning support |

See [Scope](#scope) above for why `5b`/`5b_lyrics` aren't listed here.

## Requirements

- **Python**: ≥3.10
- **PyTorch**: ≥2.13.0
- **GPU**: CUDA-capable GPU (16GB+ VRAM recommended for 1b_lyrics)
- **OS**: Linux, macOS, Windows

## Performance

Generation is intentionally slow due to autoregressive nature:
- **~5-15 seconds per second of audio** on RTX 4090 (with GPU acceleration)
- **18 seconds**: ~3-5 minutes
- **60 seconds**: ~5-15 minutes

This matches the original implementation's performance characteristics.

> **Note**: Generation speed depends on GPU, model size, and generation length. The autoregressive nature means longer generations take proportionally longer.

## Parity Verification

**jukebox-infer has been rigorously verified to produce 100% identical VQ-VAE features compared to the original OpenAI Jukebox.**

### Test Results

| Metric | Result |
|--------|--------|
| **max \|Δ\|** | 0.000000e+00 |
| **mean \|Δ\|** | 0.000000e+00 |
| **Feature shape** | (1, 6146) - identical |
| **Feature range** | [8, 2035] - identical |
| **Parity status** | **100% VERIFIED** |

### Testing Methodology

Parity was verified using:
- Multiple audio durations (5s, 20s)
- Identical official OpenAI checkpoints
- Rigorous numerical comparison (rtol=1e-4, atol=1e-6)
- Both CPU and GPU modes tested

**For full details, see [PARITY_VERIFICATION.md](docs/PARITY_VERIFICATION.md)**

## Project Structure

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
│   ├── PARITY_VERIFICATION.md      # 100% parity proof
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

## What's new

See [CHANGELOG.md](CHANGELOG.md) for the full version history. Current
release highlights:

- **v0.1.1**: Fixed the installed `jukebox-infer` console script, which
  previously failed on every install with "Error: quick_infer.py not
  found"; removed a dead `apex` import remnant from the transformer code.
- **v0.1.0**: Initial release — clean inference-only implementation
  extracted from OpenAI Jukebox.

---

## What this project will NEVER bundle

Checkpoints (~6.2GB for `1b_lyrics`) are **never committed to this
repository or bundled in the PyPI package.** They are downloaded on demand,
directly from OpenAI's original checkpoint host
(`https://openaipublic.azureedge.net/`), the first time you construct a
`Jukebox` model or run the CLI without `--help`:

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

This will not change: keeping multi-gigabyte weights out of the repo and
out of the wheel is a permanent constraint, not a temporary limitation.

---

## Limitations

- **Inference only** - No training capabilities
- **Single GPU** - No distributed inference
- **Slow generation** - Autoregressive model, ~5-15 seconds per second of audio
- **Minimum duration** - 1b_lyrics requires 17.84-600 seconds
- **Large checkpoints** - ~6.2GB download required

---

## Development

We welcome contributions! Please:

1. Follow the code style (ruff/black)
2. Add tests for new features
3. Update documentation ([CLAUDE.md](CLAUDE.md) has the module layout and
   file-header conventions)
4. Submit PRs with clear descriptions

```bash
# Install dependencies with UV
uv sync

# Run the test suite (import/CLI/hparams smoke tests, no network/GPU/weights)
uv run --with pytest python -m pytest -q

# Checkpoint URL liveness (hits the network; deselected by default)
uv run --with pytest python -m pytest -m network tests/test_weights_liveness.py -v

# Run quick inference script
uv run python quick_infer.py

# Format and lint code
uv run ruff format . && uv run ruff check .
```

See [CLAUDE.md](CLAUDE.md) for the full module layout, file-header
convention, and verification commands.

---

## License

**MIT License** (same as original Jukebox)

Copyright (c) 2020 OpenAI (Original Jukebox)
Copyright (c) 2025 (Jukebox-Infer modifications)

See [LICENSE](LICENSE) for details.

This project includes code adapted from [OpenAI Jukebox](https://github.com/openai/jukebox) (MIT License, Copyright 2020 OpenAI).

---

## Support

For issues and questions:
- **GitHub Issues**: [github.com/openmirlab/jukebox-infer/issues](https://github.com/openmirlab/jukebox-infer/issues)
- **Documentation**: `docs/`
- **Examples**: `examples/`

---

**Made with care for the ML community, on the shoulders of OpenAI and the Jukebox authors.**
# Lifecycle API

`JukeboxSession` is the explicit lifecycle facade. Call `load()` before
`infer()`; `release()` drops live model memory while retaining the checkpoint
cache, and `close()` permanently closes the session. `load()` is idempotent
for the same options, and `cache_info()` inspects the same checkpoint paths
without downloading them. Devices accept `cpu`, `cuda`, or `cuda:N`; explicit
CUDA requests fail early when unavailable. The legacy `generate()` and
`generate_from_audio()` methods remain available and lazy for compatibility.

Checkpoint URLs and cache paths are owned at runtime by the packaged
`jukebox_infer/config/checkpoints.toml`; the package never bundles weights.
