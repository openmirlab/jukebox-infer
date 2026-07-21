# jukebox-infer -- CLAUDE.md

## Scope

jukebox-infer is an inference-only repackaging of [OpenAI Jukebox](https://github.com/openai/jukebox)
(Dhariwal et al., 2020) -- no training code, no MPI/distributed dependencies,
no apex. It exposes checkpoint download + model construction + ancestral/
primed sampling behind a single high-level `Jukebox` class (`jukebox_infer.api`)
and a `jukebox-infer` console script. Only the `1b_lyrics` model
(~6.2GB checkpoints, requires a CUDA GPU with 16GB+ VRAM for real generation)
is exercised by any documented workflow today; `5b`/`5b_lyrics` entries exist
in `hparams.py`'s registry but are unused/untested. See README.md for the
public API, CLI usage, and model table.

GPU support is load-bearing: `Jukebox(device=...)` accepts `cpu`, `cuda`, and
`cuda:N`, validates explicit CUDA availability/indexes, and forwards the
resolved value through construction and sampling unchanged.

The package root uses lazy exports: importing metadata, hparams, or a
SheetSage integration does not eagerly construct the API/audio stack. This is
an import-cost optimization only; all documented root exports and standalone
generation behavior remain available when accessed.

## Module layout

- `jukebox_infer/api.py` -- the `Jukebox` class: `.load()` / `.generate()` /
  `.generate_from_audio()`, the only integration point most callers need.
- `jukebox_infer/quick_infer.py` -- CLI argument parsing + the ancestral/
  continuation drive loop. Backs both the installed `jukebox-infer` console
  script (via `cli.py`) and the repo-root `quick_infer.py` shim that
  README workflows invoke directly (`python quick_infer.py ...`).
- `jukebox_infer/cli.py` -- thin re-export: `from jukebox_infer.quick_infer
  import main`. Registered as `[project.scripts] jukebox-infer` in
  pyproject.toml.
- `jukebox_infer/make_models.py` -- checkpoint download + VQ-VAE/prior
  construction (`make_model`, `download_checkpoints`).
- `jukebox_infer/sample.py` -- `ancestral_sample` / `primed_sample`: the
  actual per-level autoregressive generation loops.
- `jukebox_infer/hparams.py` -- named hparam sets (`HPARAMS_REGISTRY`) for
  the VQ-VAE and each prior level, including each entry's checkpoint URL
  (`restore_vqvae` / `restore_prior`, all under
  `https://openaipublic.azureedge.net/`).
- `jukebox_infer/config/checkpoints.toml` -- the production catalog for every
  official runtime URL and its cache-path resolver. Unknown upstream hashes
  are recorded as `integrity = "unavailable"`; JSON is parity-only.
- `jukebox_infer/data/checkpoints.json` -- provenance registry (url / approx
  size / shared-by) mirroring the URLs embedded in hparams.py, consulted by
  `tools/check_weights_liveness.py`. No sha256 is recorded: computing one
  requires downloading the full multi-GB checkpoint, which is out of scope
  for routine maintenance passes (see Verification below).
- `jukebox_infer/vqvae/`, `jukebox_infer/prior/`, `jukebox_infer/transformer/`
  -- model architecture, unchanged from upstream Jukebox except for the
  dead `apex.normalization.FusedLayerNorm` import removed from
  `transformer/ops.py` (the plain `torch.nn.LayerNorm` fallback was always
  what ran in this inference-only package anyway).

## File-top header convention

Every load-bearing module starts with a header of this shape (as the module
docstring):

```python
"""One-line title.

2-3 sentences: what this file is for and *why* it exists this way -- the
design constraint or decision it embodies, not just a restatement of the
code.

Reads: <files/modules this one depends on>; read by: <files that depend on
this one>, where useful
"""
```

Model-architecture internals (`vqvae/`, `prior/`, `transformer/transformer.py`,
`transformer/factored_attention.py`) are intentionally left without this
treatment -- this repo's ADOPT pass touched packaging/CLI/tooling, not model
internals. `jukebox_infer/prior/conditioners.py` is the one exception: a
later bug-fix campaign (the `n_time` int-cast fix, see CHANGELOG) touched it
directly, so it got a header under the touched-file rule.

## Verification

```bash
# Import + CLI + hparams smoke tests (no network, no GPU, no weights)
uv run --with pytest python -m pytest -q

# Build + install the wheel into a scratch venv, confirm the console
# script works without downloading anything
uv build
python -m venv /tmp/scratch-venv && /tmp/scratch-venv/bin/pip install dist/*.whl
/tmp/scratch-venv/bin/jukebox-infer --help

# Checkpoint URL liveness (hits the network; deselected by default)
uv run --with pytest python -m pytest -m network tests/test_weights_liveness.py -v
# or directly:
python tools/check_weights_liveness.py
```

Real generation (`jukebox-infer` without `--help`, or `Jukebox.generate(...)`)
requires downloading ~6.2GB of checkpoints and a CUDA GPU with 16GB+ VRAM --
not exercised by CI or by routine maintenance passes on this repo.
