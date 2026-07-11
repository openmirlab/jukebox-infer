# Changelog

All notable changes to Jukebox-Infer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-07-11

### Fixed
- **`RangeEmbedding`/`LabelConditioner` crashed with `TypeError` when
  `n_time` was tainted to `numpy.float64`.** Upstream hparam arithmetic
  (e.g. `make_vqvae`'s `np.prod`-derived sample-length math) can turn
  `hps.sample_length` and everything downstream of it (`z_shapes`, `n_time`)
  into `numpy.float64` instead of a plain `int`. `RangeEmbedding.__init__`
  and `LabelConditioner.__init__` stored `n_time` verbatim, so
  `RangeEmbedding.forward`'s `t.arange(...).view(1, n_time)` later raised
  `TypeError: view(): argument 'size' failed to unpack ... got numpy.float64`
  (`torch.Tensor.view` requires a real `int`, not just an int-valued float).
  Discovered by the sheetsage-infer campaign when it replaced its vendored
  jukebox fork (which had independently fixed this) with this package; fixed
  by casting `n_time` to `int` in both constructors, in
  `jukebox_infer/prior/conditioners.py`.

## [0.1.1] - 2026-07-11

ADOPT-lite campaign (openmirlab modernization, `feat/adopt-constitution`) --
targeted fix + conformance pass, no model-architecture changes.

### Fixed
- **The installed `jukebox-infer` console script was completely broken.**
  `[project.scripts] jukebox-infer = "jukebox_infer.cli:main"` delegated to a
  repo-root `quick_infer.py` via an `os.path` hack, but that file was never
  included in the wheel (`[tool.hatch.build] include` only covers
  `jukebox_infer/**`). Every installed copy of the package failed on first
  run with "Error: quick_infer.py not found." Fixed by moving the CLI logic
  into `jukebox_infer/quick_infer.py` (packaged); `cli.py` now imports it
  directly with no path hacks. The repo-root `quick_infer.py` is kept as a
  thin shim since README workflows invoke it directly. Verified by building
  the wheel (`uv build`), installing it into a scratch venv, and running
  `jukebox-infer --help` -- clean exit, no network access, no checkpoint
  download (the heavy `Jukebox` import is deferred until after argument
  parsing returns).

### Removed
- Dead `apex.normalization.FusedLayerNorm` try-import in
  `transformer/ops.py`. Apex is a training-era compiled CUDA extension, not
  installable from PyPI for inference; the plain `torch.nn.LayerNorm`
  fallback was always the only path that actually ran in this
  inference-only package.

### Added
- `tests/`: import smoke tests (package + key submodules import cleanly,
  no GPU/network), CLI entry-point tests (`jukebox-infer --help` via
  subprocess), and hparams registry sanity tests. `[tool.pytest.ini_options]`
  previously pointed `testpaths` at a nonexistent `tests/` directory.
- `jukebox_infer/data/checkpoints.json`: url / approx-size / shared-by
  provenance for every checkpoint referenced in `hparams.py`'s
  `HPARAMS_REGISTRY` (all hosted on `openaipublic.azureedge.net`). sha256
  is intentionally not recorded -- computing it requires downloading the
  full multi-GB checkpoint, out of scope for this pass.
- `tools/check_weights_liveness.py` + `tests/test_weights_liveness.py`: a
  `pytest.mark.network`-gated liveness check (deselected by default; run
  with `pytest -m network`) that HEADs every checkpoint URL.
- `jukebox_infer/__about__.py`: single-sourced version, read by hatchling's
  `[tool.hatch.version]` and re-exported from `__init__.py` (previously
  hardcoded separately in both `pyproject.toml` and `__init__.py`).
- A `test` job in `.github/workflows/publish.yml`, now required (`needs:
  [test]`) before `publish` runs.
- Repo `CLAUDE.md`: scope, module layout, file-top header convention, and
  verification commands.
- File-top headers (title + design rationale + Reads/read-by) on
  `hparams.py`, `cli.py`, `quick_infer.py`, `make_models.py`, `sample.py`,
  and `transformer/ops.py`. Model-architecture internals (`vqvae/`,
  `prior/`, `transformer/transformer.py`, `factored_attention.py`) were
  intentionally left untouched.

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
