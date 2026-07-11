#!/usr/bin/env python3
"""Standalone entry point for `python quick_infer.py`, as documented in README.md.

The real argument-parsing and generation logic lives in
`jukebox_infer/quick_infer.py` (packaged, so it also backs the installed
`jukebox-infer` console script via jukebox_infer/cli.py). This repo-root file
is kept only because README workflows invoke it directly; it does no work of
its own.

Usage:
    python quick_infer.py [--artist ARTIST] [--genre GENRE] [--duration SECONDS] [--output OUTPUT.wav]

Examples:
    # Basic generation (default: 20 seconds, The Beatles, Rock)
    python quick_infer.py

    # Custom artist and genre
    python quick_infer.py --artist "Taylor Swift" --genre "Pop"

    # Longer generation
    python quick_infer.py --duration 30 --output my_song.wav

    # Audio continuation
    python quick_infer.py --prompt input.wav --prompt-duration 5 --duration 20

Reads: jukebox_infer.quick_infer
"""

from jukebox_infer.quick_infer import main

if __name__ == "__main__":
    main()
