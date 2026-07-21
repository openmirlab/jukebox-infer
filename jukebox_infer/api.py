"""
Simple API for Jukebox inference.
"""

import os
from typing import Optional

import numpy as np
import torch

from jukebox_infer.hparams import Hyperparams
from jukebox_infer.make_models import download_checkpoints, make_model
from jukebox_infer.sample import ancestral_sample, load_prompts, primed_sample
from jukebox_infer.utils.audio_utils import load_audio, save_wav


def resolve_device(device: Optional[str] = None) -> str:
    """
    Resolve a requested device string to a concrete "cuda"/"cpu" value.

    Args:
        device: "cuda", "cpu", "auto", or None. "auto" and None (unset) both
            auto-detect: "cuda" if a GPU is available, otherwise "cpu".
            Explicit "cuda"/"cpu" pass through unchanged.

    Returns:
        "cuda" or "cpu"
    """
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', or 'cuda:N'")
    suffix = device[4:]
    if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit()):
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', or 'cuda:N'")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was explicitly requested ({device}) but is unavailable")
    if suffix and int(suffix[1:]) >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {suffix[1:]} is unavailable")
    return device


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # For full determinism (may slow down slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Jukebox:
    """
    Simple API for Jukebox music generation.

    Example:
        >>> model = Jukebox("5b_lyrics", device="cuda")
        >>> audio = model.generate(artist="The Beatles", genre="Rock", duration=20)
    """

    def __init__(self, model_name="5b_lyrics", device=None):
        """
        Initialize Jukebox model.

        Args:
            model_name: Model to use ("1b_lyrics", "5b", or "5b_lyrics")
            device: Device to run on ("cuda", "cpu", or "auto"). Defaults to
                None, which auto-detects the same way "auto" does: "cuda" if
                available, otherwise "cpu".
        """
        self.model_name = model_name
        self.device = resolve_device(device)
        self.vqvae = None
        self.priors = None
        self._closed = False
        self._loaded = False
        self._load_args = None

    def load(self, sample_length_in_seconds=20, n_samples=1, auto_download=True):
        """
        Load the model (downloads checkpoints automatically if needed).

        Args:
            sample_length_in_seconds: Length of audio to generate
            n_samples: Number of samples to generate in parallel
            auto_download: If True, automatically download missing checkpoints
        """
        if self._closed:
            raise RuntimeError("Jukebox session is closed; create a new session")
        args = (sample_length_in_seconds, n_samples, auto_download)
        if self._loaded and self._load_args == args:
            return self
        if self._loaded:
            raise RuntimeError("Jukebox is already loaded with different load options; release() first")
        hps = Hyperparams(
            sample_length_in_seconds=sample_length_in_seconds,
            total_sample_length_in_seconds=sample_length_in_seconds,
            sr=44100,
            n_samples=n_samples,
            hop_fraction=[0.5, 0.5, 0.125]
        )

        print(f"Loading {self.model_name}...")
        if auto_download:
            print("Note: Missing checkpoints will be downloaded automatically.")
        self.vqvae, self.priors = make_model(self.model_name, self.device, hps, auto_download=auto_download)
        self.hps = hps
        self._loaded = True
        self._load_args = args
        print("✓ Model loaded successfully")
        return self

    def generate(
        self,
        artist="",
        genre="",
        lyrics="",
        duration_seconds=20,
        temperature=0.99,
        output_path=None,
        seed: Optional[int] = None,
    ):
        """
        Generate music from scratch.

        Args:
            artist: Artist name to condition on
            genre: Genre to condition on
            lyrics: Lyrics to condition on (for lyrics models)
            duration_seconds: Duration of audio to generate
            temperature: Sampling temperature (default 0.99)
            output_path: Where to save audio (optional)
            seed: Random seed for reproducibility (optional)

        Returns:
            numpy array of audio samples
        """
        if seed is not None:
            set_seed(seed)

        if self.vqvae is None:
            self.load(sample_length_in_seconds=duration_seconds)

        # Create labels
        metas = [dict(
            artist=artist,
            genre=genre,
            lyrics=lyrics,
            total_length=duration_seconds * self.hps.sr,
            offset=0
        )]
        labels = [prior.labeller.get_batch_labels(metas, self.device)
                 for prior in self.priors]

        # Sampling kwargs - optimized for GPU
        # Larger batch sizes = better GPU utilization
        chunk_size = 32 if self.model_name == '1b_lyrics' else 16
        max_batch_size = 32 if self.model_name == '1b_lyrics' else 16
        sampling_kwargs = [
            dict(temp=temperature, fp16=True, chunk_size=64, max_batch_size=32),
            dict(temp=temperature, fp16=True, chunk_size=64, max_batch_size=32),
            dict(temp=temperature, fp16=True, chunk_size=chunk_size, max_batch_size=max_batch_size)
        ]

        # Generate
        print("Generating music...")
        # Convert device string to torch.device if needed
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        with torch.no_grad():
            zs = ancestral_sample(labels, sampling_kwargs, self.priors, self.hps, device=device)

        # Decode to audio
        # zs is a list with one tensor per level, decode needs all levels from start_level to end_level
        print("Decoding to audio...")
        audio = self.priors[-1].decode(zs, start_level=0, bs_chunks=1)
        audio = audio.cpu().numpy()

        if output_path:
            # Handle both file paths and directory paths
            if os.path.splitext(output_path)[1]:  # Has extension, treat as file
                output_dir = os.path.dirname(output_path) or "."
                os.makedirs(output_dir, exist_ok=True)
                # Save directly using soundfile for single file
                import soundfile
                aud_clipped = torch.clamp(torch.from_numpy(audio), -1, 1).numpy()
                soundfile.write(output_path, aud_clipped[0], samplerate=self.hps.sr, format='wav')
                print(f"Saved to {output_path}")
            else:  # Directory path
                os.makedirs(output_path, exist_ok=True)
                save_wav(output_path, audio, self.hps.sr)
                print(f"Saved to {output_path}/item_0.wav")

        return audio

    @property
    def status(self):
        if self._closed:
            return "closed"
        return "ready" if self._loaded and self.vqvae is not None else "new"

    def release(self):
        """Release live model objects while retaining the on-disk checkpoint cache."""
        if self._closed:
            return
        self.vqvae = None
        self.priors = None
        self._loaded = False
        self._load_args = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self):
        if not self._closed:
            self.release()
            self._closed = True

    def cache_info(self):
        from jukebox_infer.config import checkpoint_cache_info
        return checkpoint_cache_info(self.model_name)

    def infer(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Jukebox session is closed; create a new session")
        if self.status != "ready":
            raise RuntimeError("Jukebox session is not ready; call load() first")
        return self.generate(*args, **kwargs)

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        self.close()


    def generate_from_audio(
        self,
        prompt_audio,
        prompt_duration=12,
        total_duration=30,
        temperature=0.99,
        output_path=None,
        seed: Optional[int] = None,
    ):
        """
        Generate music continuation from an audio prompt.

        Args:
            prompt_audio: Path to audio file or numpy array
            prompt_duration: How many seconds of prompt to use
            total_duration: Total duration to generate
            temperature: Sampling temperature (default 0.99)
            output_path: Where to save audio (optional)
            seed: Random seed for reproducibility (optional)

        Returns:
            numpy array of audio samples
        """
        if seed is not None:
            set_seed(seed)

        if self.vqvae is None:
            self.load(sample_length_in_seconds=total_duration)

        # Load prompt
        if isinstance(prompt_audio, str):
            x = load_prompts([prompt_audio],
                           prompt_duration * self.hps.sr,
                           self.hps,
                           device=self.device)
        else:
            x = torch.from_numpy(prompt_audio).unsqueeze(0).to(self.device)

        # Create empty labels
        metas = [dict(artist="", genre="", lyrics="",
                     total_length=total_duration * self.hps.sr, offset=0)]
        labels = [prior.labeller.get_batch_labels(metas, self.device)
                 for prior in self.priors]

        # Sampling kwargs - optimized for GPU
        chunk_size = 32 if self.model_name == '1b_lyrics' else 16
        max_batch_size = 32 if self.model_name == '1b_lyrics' else 16
        sampling_kwargs = [
            dict(temp=temperature, fp16=True, chunk_size=64, max_batch_size=32),
            dict(temp=temperature, fp16=True, chunk_size=64, max_batch_size=32),
            dict(temp=temperature, fp16=True, chunk_size=chunk_size, max_batch_size=max_batch_size)
        ]

        # Generate
        print("Generating continuation...")
        # Convert device string to torch.device if needed
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        with torch.no_grad():
            zs = primed_sample(x, labels, sampling_kwargs, self.priors, self.hps, device=device)

        # Decode to audio
        # zs is a list with one tensor per level, decode needs all levels from start_level to end_level
        print("Decoding to audio...")
        audio = self.priors[-1].decode(zs, start_level=0, bs_chunks=1)
        audio = audio.cpu().numpy()

        if output_path:
            # Handle both file paths and directory paths
            if os.path.splitext(output_path)[1]:  # Has extension, treat as file
                output_dir = os.path.dirname(output_path) or "."
                os.makedirs(output_dir, exist_ok=True)
                # Save directly using soundfile for single file
                import soundfile
                aud_clipped = torch.clamp(torch.from_numpy(audio), -1, 1).numpy()
                soundfile.write(output_path, aud_clipped[0], samplerate=self.hps.sr, format='wav')
                print(f"Saved to {output_path}")
            else:  # Directory path
                os.makedirs(output_path, exist_ok=True)
                save_wav(output_path, audio, self.hps.sr)
                print(f"Saved to {output_path}/item_0.wav")

        return audio


JukeboxSession = Jukebox
