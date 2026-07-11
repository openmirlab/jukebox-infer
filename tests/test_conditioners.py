"""RangeEmbedding / LabelConditioner int-cast regression test, no network, no GPU.

Reproduces the bug the sheetsage-infer campaign hit when it swapped its
vendored Jukebox fork for this package: upstream arithmetic (e.g.
`make_vqvae`'s `np.prod(...)`-derived sample-length math) can taint hparams
like `hps.sample_length` into `numpy.float64` instead of a plain `int`. That
taint flows into `RangeEmbedding`/`LabelConditioner`'s `n_time`, and
`RangeEmbedding.forward`'s `t.arange(...).view(1, n_time)` raises `TypeError`
because `torch.Tensor.view` requires an actual int (no `__index__` shim for
`numpy.float64`). This module asserts both classes store `n_time` as `int`
regardless of the input dtype, so a numpy-tainted caller never reaches the
`.view()` call with a non-int size.

Reads: jukebox_infer.prior.conditioners (module under test)
"""

import numpy as np
import pytest
import torch as t

from jukebox_infer.prior.conditioners import LabelConditioner, RangeEmbedding


def test_range_embedding_forward_with_numpy_float64_n_time():
    """RangeEmbedding must survive a numpy.float64-tainted n_time end to end."""
    tainted_n_time = np.prod([8], dtype=np.int64) / 1.0  # mirrors make_vqvae's np.prod taint
    assert isinstance(tainted_n_time, np.float64)

    emb = RangeEmbedding(
        tainted_n_time, bins=16, range=(0.0, 1.0), out_width=4, init_scale=1.0
    )
    assert isinstance(emb.n_time, int), f"Expected int n_time, got {type(emb.n_time)}"

    pos_start = t.zeros(1, 1)
    pos_end = t.ones(1, 1)
    out = emb(pos_start, pos_end)
    assert out.shape == (1, 8, 4)


def test_label_conditioner_stores_int_n_time_with_numpy_float64_input():
    """LabelConditioner has the same untyped-n_time sibling the fork also cast."""
    tainted_n_time = np.prod([8], dtype=np.int64) / 1.0
    assert isinstance(tainted_n_time, np.float64)

    cond = LabelConditioner(
        y_bins=(4, 4),
        t_bins=16,
        sr=44100,
        min_duration=1.0,
        max_duration=2.0,
        n_time=tainted_n_time,
        out_width=4,
        init_scale=1.0,
        max_bow_genre_size=1,
        include_time_signal=True,
    )
    assert isinstance(cond.n_time, int), f"Expected int n_time, got {type(cond.n_time)}"
