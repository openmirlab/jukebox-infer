#!/usr/bin/env python3
"""Weights-liveness check -- HEADs every checkpoint URL in data/checkpoints.json.

Mirrors maest-infer's tools/check_weights_liveness.py convention: all of
jukebox-infer's checkpoints are third-party OpenAI Jukebox releases hosted
on a single Azure mirror (openaipublic.azureedge.net) with no mirror of our
own, so a dead upstream host breaks every model load with no earlier
warning. This script walks checkpoints.json and HEADs each URL. Run
manually (`python tools/check_weights_liveness.py`) or via `pytest -m
network tests/test_weights_liveness.py` -- neither runs in the default test
suite since both need real network access.

Reads: jukebox_infer/data/checkpoints.json
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

TIMEOUT_SECONDS = 15
CHECKPOINTS_JSON = (
    Path(__file__).resolve().parent.parent / "jukebox_infer" / "data" / "checkpoints.json"
)


def iter_registry_urls() -> Iterator[tuple[str, str]]:
    """Yield (label, url) for every checkpoint recorded in checkpoints.json."""
    with open(CHECKPOINTS_JSON) as f:
        data = json.load(f)
    for label, entry in data.get("checkpoints", {}).items():
        yield label, entry["url"]


def check_url(url: str, timeout: float = TIMEOUT_SECONDS) -> tuple[bool, str]:
    """HEAD a URL, falling back to a ranged GET if the host doesn't support HEAD."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code in (405, 501):
            # Host doesn't support HEAD; retry with a ranged GET.
            ranged = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            try:
                with urllib.request.urlopen(ranged, timeout=timeout) as resp:
                    return 200 <= resp.status < 400, f"HTTP {resp.status}"
            except (urllib.error.HTTPError, urllib.error.URLError) as exc2:
                return False, f"error: {exc2}"
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, f"error: {exc}"


def main() -> int:
    failures = []
    for label, url in iter_registry_urls():
        ok, detail = check_url(url)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {label}: {url} -> {detail}")
        if not ok:
            failures.append((label, url, detail))

    print()
    if failures:
        print(f"{len(failures)} URL(s) failed liveness check:")
        for label, url, detail in failures:
            print(f"  - {label}: {url} ({detail})")
        return 1

    print("All registry URLs are live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
