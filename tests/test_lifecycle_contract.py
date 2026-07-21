"""Offline device and lifecycle contracts for the public Jukebox facade."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from jukebox_infer import api
from jukebox_infer.api import Jukebox, resolve_device


def test_device_validation_and_cuda_index_forwarding():
    calls = []

    def fake_make_model(name, device, hps, auto_download):
        calls.append((name, device, auto_download))
        return object(), []

    with patch.object(api.torch.cuda, "is_available", return_value=True), \
         patch.object(api.torch.cuda, "device_count", return_value=2), \
         patch.object(api, "make_model", fake_make_model):
        assert resolve_device(None) == "cuda"
        assert resolve_device("auto") == "cuda"
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda:1") == "cuda:1"
        Jukebox("1b_lyrics", device="cuda:1").load(auto_download=False)
    assert calls == [("1b_lyrics", "cuda:1", False)]

    with patch.object(api.torch.cuda, "is_available", return_value=False):
        assert resolve_device(None) == "cpu"
        with pytest.raises(RuntimeError, match="unavailable"):
            resolve_device("cuda")
    with pytest.raises(ValueError, match="device must"):
        resolve_device("mps")


def test_load_is_idempotent_and_reloads_only_after_release(monkeypatch):
    calls = []
    monkeypatch.setattr(
        api,
        "make_model",
        lambda name, device, hps, auto_download: calls.append((name, device)) or (object(), []),
    )
    session = Jukebox("1b_lyrics", device="cpu")
    assert session.load(auto_download=False) is session
    assert session.load(auto_download=False) is session
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="different load options"):
        session.load(sample_length_in_seconds=30, auto_download=False)
    session.release()
    session.load(auto_download=False)
    assert len(calls) == 2


def test_infer_is_ready_only_and_close_is_terminal(monkeypatch):
    session = Jukebox(device="cpu")
    with pytest.raises(RuntimeError, match="not ready"):
        session.infer()
    monkeypatch.setattr(api, "make_model", lambda *args, **kwargs: (object(), []))
    session.load(auto_download=False)
    session.close()
    session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load(auto_download=False)
    with pytest.raises(RuntimeError, match="closed"):
        session.infer()


def test_cache_info_delegates_to_existing_checkpoint_resolver(monkeypatch):
    expected = {"model": "1b_lyrics", "cached": False}
    monkeypatch.setattr("jukebox_infer.config.checkpoint_cache_info", lambda name: expected)
    assert Jukebox("1b_lyrics", device="cpu").cache_info() is expected
