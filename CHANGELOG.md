# Changelog

All notable changes to Jukebox-Infer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## CI matrix + dependency floor refresh (2026-07-12, branch `fix/ci-matrix`)

An org audit found this repo tested only Python 3.10, and only inside
`publish.yml`'s release-gate job -- there was no dedicated `test.yml` running
on every push/PR, even though `pyproject.toml` classifiers claimed Python
3.10-3.12 support. This change builds that CI and confirms every claimed
Python version actually passes the 27-test suite, plus refreshes stale
dependency floors. Dependabot: 0 open alerts (reconfirmed via
`gh api repos/openmirlab/jukebox-infer/dependabot/alerts`), no security fixes
needed.

### Added
- **`.github/workflows/test.yml`**: a `test` job running the full `pytest`
  suite across a Python 3.10/3.11/3.12/3.13 matrix via `uv` (`uv sync --extra
  dev --python <version>` -- this repo declares dev deps under
  `[project.optional-dependencies].dev`, not a uv dependency-group, so the
  bare `uv sync` in `publish.yml` would silently skip pytest; `--extra dev`
  is required and was verified empirically), plus a `build` job (`needs:
  [test]`) doing the wheel-from-sdist build (`python -m build`), a clean-venv
  install, and an import smoke test touching `Jukebox` and `__version__`
  (org constitution article 7). All four Python versions were verified green
  *before* being added to the matrix -- none were excluded. Python 3.13 was
  the one at risk (torch historically lagged on cp313 wheels) but torch
  2.13.0 ships `cp310`/`cp311`/`cp312`/`cp313` manylinux wheels, confirmed via
  the PyPI file listing, and the suite passed under all four.
- `Programming Language :: Python :: 3.13` classifier, now that 3.13 is
  verified green.

### Changed
- Stale dependency floors raised, each re-verified green across the full
  Python 3.10-3.13 matrix after bumping:
  - `torch>=2.7.0` -> `>=2.13.0` (current PyPI latest stable; has cp310-cp313
    wheels).
  - `numpy>=1.21.0` -> `>=2.2.6` (not the current numpy latest, 2.5.1: numpy
    `>=2.3` requires Python>=3.11 and `>=2.5` requires Python>=3.12, both of
    which would break this repo's `>=3.10` floor. `2.2.6` is the newest 2.x
    release with a Python 3.10 wheel -- verified via the PyPI JSON API's
    `requires_python` field for 2.2.6 (`>=3.10`), 2.3.4 (`>=3.11`), and 2.5.1
    (`>=3.12`). Same trap and same resolution as larsnet-infer's CHANGELOG.)
  - `tqdm>=4.0.0` -> `>=4.68.4` (current PyPI latest; low-risk, no Python
    floor conflict -- `requires_python >=3.8`).
  - `librosa`, `soundfile`, `unidecode` floors left untouched -- not flagged
    stale by the prior audit and no incompatibility surfaced while testing
    the matrix.
- `uv.lock` regenerated (`uv lock`, 79 packages resolved) after the floor
  bumps, then re-synced and re-tested per Python version with `uv sync
  --locked` to confirm the lockfile actually satisfies all four versions.
- `pyproject.toml` description and README PyTorch references updated from
  `2.7+` to `2.13+` to match the new floor (badge, requirements section,
  "What we maintain" section).
- Test count confirmed unchanged across every step of this change: **27
  passed / 6 deselected** (the 6 are `network`-marked, deselected by the
  `-m not network` addopt), identical on Python 3.10, 3.11, 3.12, and 3.13,
  both before and after the floor bumps.

`publish.yml`'s own `test` job is left untouched (still Python-3.10-only) --
that is a deliberate release-gate simplicity per the existing convention,
not something this change needed to fix; `test.yml` is now the workflow that
actually exercises the full matrix on every push/PR.

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
