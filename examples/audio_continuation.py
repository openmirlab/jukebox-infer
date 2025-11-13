"""
Example of continuing from an audio prompt

This example shows how to generate a continuation from an existing audio file.
The model will extend the provided audio prompt.
"""

from jukebox_infer import Jukebox

# Initialize model (checkpoints auto-download on first use)
model = Jukebox(model_name="1b_lyrics", device="cuda")

# Load model
model.load(sample_length_in_seconds=30, n_samples=1)

# Generate continuation from audio file
print("Generating continuation... (this will take a while)")
audio = model.generate_from_audio(
    prompt_audio="input.wav",
    prompt_duration=12,
    total_duration=30,
    output_path="continuation.wav"
)

print(f"✓ Generated continuation saved to continuation.wav")
print(f"  Shape: {audio.shape}")
