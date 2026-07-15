"""Import smoke tests: the package and its key submodules import cleanly, no network.

Guards against the class of bug this campaign fixed for the CLI (a code path
that only breaks once installed from a wheel, not from the repo checkout) by
asserting the same things at the module level: `import jukebox_infer` works,
`__version__` is a real string, and the submodules the CLI/API chain touches
(quick_infer, cli, make_models, sample, hparams, transformer.ops) import
without requiring a GPU or any network access -- checkpoint downloads only
happen when a function is actually called, never at import time.

Reads: jukebox_infer (package under test)
"""

import importlib
import subprocess
import sys

import pytest


def test_package_imports():
    import jukebox_infer

    assert jukebox_infer is not None


def test_version_is_nonempty_string():
    import jukebox_infer

    assert isinstance(jukebox_infer.__version__, str)
    assert jukebox_infer.__version__ != ""


def test_public_api_names_exported():
    import jukebox_infer

    assert hasattr(jukebox_infer, "Jukebox")
    assert hasattr(jukebox_infer, "set_seed")
    assert hasattr(jukebox_infer, "download_checkpoints")


def test_package_import_defers_heavy_api_modules():
    """Metadata/hparams imports must not eagerly import the API/audio stack."""
    code = (
        "import sys; import jukebox_infer; import jukebox_infer.hparams; "
        "assert 'jukebox_infer.api' not in sys.modules; "
        "assert 'jukebox_infer.make_models' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "module_name",
    [
        "jukebox_infer",
        "jukebox_infer.__about__",
        "jukebox_infer.api",
        "jukebox_infer.cli",
        "jukebox_infer.quick_infer",
        "jukebox_infer.make_models",
        "jukebox_infer.sample",
        "jukebox_infer.hparams",
        "jukebox_infer.transformer.ops",
        "jukebox_infer.transformer.transformer",
        "jukebox_infer.transformer.factored_attention",
        "jukebox_infer.vqvae.vqvae",
        "jukebox_infer.prior.prior",
        "jukebox_infer.utils.remote_utils",
    ],
)
def test_submodule_imports_cleanly(module_name):
    module = importlib.import_module(module_name)
    assert module is not None
