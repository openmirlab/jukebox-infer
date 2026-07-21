"""Device-resolution regression tests: auto-detect on unset/"auto", no GPU, no network.

Bug: `Jukebox.__init__(self, model_name="5b_lyrics", device="cuda")` hardcoded
a literal "cuda" default with no auto-detection at all -- worse than sibling
packages in this org, which at least auto-detect on `None`. A CPU-only
machine calling `Jukebox()` with no args loaded successfully but crashed at
the first `.to("cuda")` call inside `load()`/`generate()`. `resolve_device()`
now centralizes that decision: the unset default and the explicit "auto"
sentinel both auto-detect via `torch.cuda.is_available()`; explicit "cuda"/
"cpu" pass through unchanged (those are deliberate caller choices, and the
call sites that apply an already-resolved device -- api.py's `.to(device)`
calls -- are untouched by this fix).

Reads: jukebox_infer.api (resolve_device, Jukebox), jukebox_infer.quick_infer
(build_parser)
"""

from unittest.mock import patch

from jukebox_infer.api import Jukebox, resolve_device


def test_resolve_device_auto_detects_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device(None) == "cuda"
        assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_detects_cpu_when_unavailable():
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_device(None) == "cpu"
        assert resolve_device("auto") == "cpu"


def test_resolve_device_passes_through_explicit_values_unchanged():
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda") == "cuda"
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_device("cpu") == "cpu"


def test_jukebox_default_device_auto_detects_without_gpu():
    with patch("torch.cuda.is_available", return_value=False):
        model = Jukebox()
        assert model.device == "cpu"


def test_jukebox_default_device_auto_detects_with_gpu():
    with patch("torch.cuda.is_available", return_value=True):
        model = Jukebox()
        assert model.device == "cuda"


def test_jukebox_explicit_auto_device():
    with patch("torch.cuda.is_available", return_value=False):
        model = Jukebox(device="auto")
        assert model.device == "cpu"


def test_jukebox_explicit_device_passes_through_unchanged():
    with patch("torch.cuda.is_available", return_value=True):
        assert Jukebox(device="cpu").device == "cpu"
    with patch("torch.cuda.is_available", return_value=False):
        import pytest
        with pytest.raises(RuntimeError):
            Jukebox(device="cuda")


def test_quick_infer_device_cli_choices_include_auto_and_default_to_it():
    from jukebox_infer.quick_infer import build_parser

    parser = build_parser()
    device_action = next(a for a in parser._actions if a.dest == "device")
    assert set(device_action.choices) == {"cuda", "cpu", "auto"}
    assert device_action.default == "auto"
