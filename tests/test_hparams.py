"""Hparams registry sanity tests: named hparam sets compose correctly, no checkpoint I/O.

`setup_hparams` is the one function every model-construction path
(`make_models.make_vqvae` / `make_prior`) depends on to resolve a
comma-separated list of `HPARAMS_REGISTRY` names into a single `Hyperparams`
dict. These tests only exercise dict composition -- no checkpoint is
downloaded or loaded.

Reads: jukebox_infer.hparams
"""

import pytest

from jukebox_infer.hparams import HPARAMS_REGISTRY, REMOTE_PREFIX, setup_hparams


def test_registry_has_expected_entries():
    for name in (
        "vqvae",
        "upsampler_level_0",
        "upsampler_level_1",
        "prior_5b",
        "prior_5b_lyrics",
        "prior_1b_lyrics",
    ):
        assert name in HPARAMS_REGISTRY


def test_setup_hparams_single_name():
    hps = setup_hparams("vqvae", {})
    assert hps.restore_vqvae.startswith(REMOTE_PREFIX)
    assert hps.levels == 3


def test_setup_hparams_composes_multiple_names():
    hps = setup_hparams("upsampler_level_0", {})
    assert hps.level == 0
    assert hps.restore_prior.startswith(REMOTE_PREFIX)


def test_setup_hparams_kwargs_override():
    hps = setup_hparams("vqvae", {"sample_length": 12345})
    assert hps.sample_length == 12345


def test_setup_hparams_rejects_unknown_key():
    with pytest.raises(ValueError):
        setup_hparams("vqvae", {"totally_unknown_key": 1})
