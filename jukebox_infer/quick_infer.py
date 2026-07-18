"""Quick-inference CLI logic: argument parsing plus the ancestral/continuation drive loop.

This is the real implementation behind both the `jukebox-infer` console script
(jukebox_infer/cli.py) and the repo-root `quick_infer.py` shim that README
workflows invoke directly (`python quick_infer.py ...`). It used to live only
at the repo root, which meant the installed package's console script had
nothing to delegate to (repo-root scripts aren't shipped in the wheel) and
always failed with "Error: quick_infer.py not found." Moving the logic here
makes it part of the installable package; the heavy `jukebox_infer.Jukebox`
import and any checkpoint download only happen after argparse has already
returned, so `--help` stays fast and network-free.

Reads: jukebox_infer (Jukebox API, imported lazily inside main()); read by:
jukebox_infer.cli, repo-root quick_infer.py
"""

import argparse
import os
import sys

CLI_EPILOG = """\
Usage:
    jukebox-infer [--artist ARTIST] [--genre GENRE] [--duration SECONDS] [--output OUTPUT.wav]

Examples:
    # Basic generation (default: 20 seconds, The Beatles, Rock)
    jukebox-infer

    # Custom artist and genre
    jukebox-infer --artist "Taylor Swift" --genre "Pop"

    # Longer generation
    jukebox-infer --duration 30 --output my_song.wav

    # Audio continuation
    jukebox-infer --prompt input.wav --prompt-duration 5 --duration 20
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the quick-inference argument parser (no heavy imports)."""
    parser = argparse.ArgumentParser(
        description="Quick inference with Jukebox-Infer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLI_EPILOG,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="1b_lyrics",
        choices=["1b_lyrics"],
        help="Model to use (default: 1b_lyrics)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to use (default: auto, i.e. cuda if available else cpu)",
    )

    parser.add_argument(
        "--artist",
        type=str,
        default="The Beatles",
        help="Artist name for conditioning (default: The Beatles)",
    )

    parser.add_argument(
        "--genre",
        type=str,
        default="Rock",
        help="Genre for conditioning (default: Rock)",
    )

    parser.add_argument(
        "--lyrics",
        type=str,
        default="",
        help="Lyrics for conditioning (optional, only for lyrics models)",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Duration in seconds (default: 20, minimum: 18 for 1b_lyrics)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.99,
        help="Sampling temperature (default: 0.99)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output file path (default: output.wav)",
    )

    # Audio continuation options
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Path to audio file for continuation (optional)",
    )

    parser.add_argument(
        "--prompt-duration",
        type=float,
        default=5.0,
        help="Duration of prompt audio to use in seconds (default: 5.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate duration
    if args.duration < 18:
        print("⚠️  Warning: Duration must be at least 18 seconds for 1b_lyrics. Setting to 18.")
        args.duration = 18

    # Check if prompt file exists
    if args.prompt and not os.path.exists(args.prompt):
        print(f"❌ Error: Prompt file not found: {args.prompt}")
        sys.exit(1)

    print("=" * 70)
    print("JUKEBOX-INFER QUICK INFERENCE")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    if args.prompt:
        print("Mode: Audio Continuation")
        print(f"Prompt: {args.prompt} ({args.prompt_duration}s)")
    else:
        print("Mode: Ancestral Generation")
        print(f"Artist: {args.artist}")
        print(f"Genre: {args.genre}")
    print(f"Duration: {args.duration} seconds")
    print(f"Output: {args.output}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("=" * 70)

    try:
        # Heavy import (torch, checkpoint download machinery) deferred until
        # after argument parsing so `--help` never touches the network.
        from jukebox_infer import Jukebox

        print("\n[1/3] Initializing model...")
        model = Jukebox(model_name=args.model, device=args.device)

        print("[2/3] Loading model (checkpoints will auto-download if needed)...")
        model.load(sample_length_in_seconds=args.duration, n_samples=1)

        print("[3/3] Generating audio...")
        print(
            f"      Note: This will take ~{args.duration * 5}-{args.duration * 15} seconds "
            "(~5-15s per second of audio)"
        )

        if args.prompt:
            # Audio continuation
            audio = model.generate_from_audio(
                prompt_audio=args.prompt,
                prompt_duration=args.prompt_duration,
                total_duration=args.duration,
                temperature=args.temperature,
                output_path=args.output,
                seed=args.seed,
            )
        else:
            # Ancestral generation
            audio = model.generate(
                artist=args.artist,
                genre=args.genre,
                lyrics=args.lyrics,
                duration_seconds=args.duration,
                temperature=args.temperature,
                output_path=args.output,
                seed=args.seed,
            )

        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"Generated audio saved to: {args.output}")
        print(f"Audio shape: {audio.shape}")
        print("Sample rate: 44100 Hz")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
