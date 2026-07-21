"""Offline TOML runtime-catalog and legacy-JSON parity contracts."""
from __future__ import annotations

import json

import pytest

from jukebox_infer import config, make_models


def test_toml_catalog_has_complete_metadata_and_matches_legacy_url_parity():
    legacy_path = config.CONFIG_PATH.parents[1] / "data" / "checkpoints.json"
    legacy = json.loads(legacy_path.read_text())["checkpoints"]
    catalog = config.checkpoint_catalog()
    assert set(catalog) == set(legacy)
    for name, record in catalog.items():
        assert record["url"] == legacy[name]["url"]
        assert record["integrity"] == "unavailable"
        assert {"size_bytes", "license", "provenance", "source_revision", "updated"} <= set(record)


def test_model_loading_and_cache_info_share_toml_checkpoint_resolver(monkeypatch, tmp_path):
    calls = []

    def fake_resolver(url):
        calls.append(url)
        return str(tmp_path / "missing.ckpt")

    monkeypatch.setattr(config, "resolve_checkpoint_path", fake_resolver)
    with pytest.raises(FileNotFoundError):
        make_models.load_checkpoint(config.checkpoint_url("vqvae"), auto_download=False)
    config.checkpoint_cache_info("1b_lyrics")
    assert calls[0] == config.checkpoint_url("vqvae")
    assert config.checkpoint_url("vqvae") in calls[1:]


def test_toml_resolver_path_matches_legacy_remote_cache_mapping(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    url = config.checkpoint_url("prior_1b_lyrics")
    assert config.resolve_checkpoint_path(url) == str(
        tmp_path / ".cache" / "jukebox" / "models" / "1b_lyrics" / "prior_level_2.pth.tar"
    )
