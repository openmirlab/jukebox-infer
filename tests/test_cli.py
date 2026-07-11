"""CLI entry-point tests: `jukebox-infer --help` must work with no network, no GPU, no weights.

This is the regression test for the campaign's headline bug: the
`jukebox-infer` console script (jukebox_infer.cli:main) used to `os.path`-hack
its way to a repo-root `quick_infer.py` that was never shipped in the wheel,
so any installed copy of the package failed immediately with "Error:
quick_infer.py not found." That logic now lives inside the package
(jukebox_infer/quick_infer.py), so `--help` must succeed purely via module
invocation (`python -m jukebox_infer.cli --help`) without touching the
network or downloading any checkpoint -- argparse returns before the heavy
`Jukebox` import is ever reached.

Reads: jukebox_infer.cli (invoked as a subprocess module, not imported
directly, so this also exercises the real console-script code path)
"""

import subprocess
import sys


def test_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "jukebox_infer.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_cli_help_mentions_quick_inference_options():
    result = subprocess.run(
        [sys.executable, "-m", "jukebox_infer.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "--artist" in result.stdout
    assert "--genre" in result.stdout
    assert "--duration" in result.stdout


def test_cli_help_does_not_import_error_out():
    # The historical bug printed exactly this on every invocation.
    result = subprocess.run(
        [sys.executable, "-m", "jukebox_infer.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "quick_infer.py not found" not in result.stdout
    assert "quick_infer.py not found" not in result.stderr
