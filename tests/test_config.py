"""cache_info() regression tests: per-file checkpoint check, not directory existence.

Bug: `checkpoint_cache_info()` used to check only whether
`~/.cache/jukebox/models` exists as a *directory*, ignoring `model_name`
entirely. That disagreed with the real download-decision check
(`utils.remote_utils.check_file_exists`, which `make_models.load_checkpoint`
actually uses to decide whether to download) -- a shared cache directory can
exist (e.g. holding some other model's files, or a stray file) while a
*specific* model's checkpoint files are missing. These tests construct
exactly that scenario: the cache directory exists (with an unrelated file in
it) but none of "5b_lyrics"'s checkpoint files are present, so the old
directory-only check would report "exists" (and would have been read as
"cached") while the new per-file check correctly reports "not cached".

Reads: jukebox_infer.config (module under test), jukebox_infer.make_models,
jukebox_infer.hparams
"""

import os

import pytest

from jukebox_infer.config import _expected_checkpoint_paths, checkpoint_cache_info


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_cache_info_reports_not_cached_when_directory_exists_but_files_missing(fake_home):
    # Directory exists (the old check's entire signal), holding an unrelated file...
    models_dir = fake_home / ".cache" / "jukebox" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "unrelated.txt").write_text("not a checkpoint")

    info = checkpoint_cache_info("5b_lyrics")

    # ...but none of 5b_lyrics's actual checkpoint files are there. The old
    # directory-existence check would have reported this as "exists", which
    # is the wrong answer for "is 5b_lyrics cached".
    assert info["exists"] is True
    assert info["cached"] is False
    assert info["files"], "expected at least one resolved checkpoint path"
    assert set(info["missing"]) == set(info["files"])


def test_cache_info_reports_cached_when_expected_files_present(fake_home):
    expected_paths = _expected_checkpoint_paths("5b_lyrics")
    assert expected_paths, "expected at least one checkpoint file for 5b_lyrics"
    for p in expected_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"fake checkpoint bytes")  # non-zero size: check_file_exists requires it

    info = checkpoint_cache_info("5b_lyrics")

    assert info["cached"] is True
    assert info["missing"] == []


def test_cache_info_zero_byte_file_is_not_treated_as_cached(fake_home):
    """A zero-byte file (e.g. a failed/partial download) must not count as cached."""
    expected_paths = _expected_checkpoint_paths("5b_lyrics")
    for p in expected_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").close()  # zero bytes

    info = checkpoint_cache_info("5b_lyrics")

    assert info["cached"] is False
    assert set(info["missing"]) == set(expected_paths)


def test_cache_info_distinguishes_models_with_different_files():
    # 1b_lyrics and 5b_lyrics resolve to distinct prior checkpoint files, so
    # caching one must not be mistaken for caching the other.
    lyrics_5b_paths = set(_expected_checkpoint_paths("5b_lyrics"))
    lyrics_1b_paths = set(_expected_checkpoint_paths("1b_lyrics"))
    assert lyrics_5b_paths != lyrics_1b_paths


def test_cache_info_unknown_model_raises():
    with pytest.raises(ValueError):
        checkpoint_cache_info("not_a_real_model")


def test_cache_info_none_model_preserves_directory_only_shape(fake_home):
    info = checkpoint_cache_info(None)
    assert info == {
        "directory": str(fake_home / ".cache" / "jukebox" / "models"),
        "exists": False,
        "model": None,
    }
