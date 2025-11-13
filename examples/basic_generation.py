"""
Basic example of generating music with Jukebox-Infer

This example demonstrates how to use the simple API to generate music.
Checkpoints will be automatically downloaded on first use (~6.2GB for 1b_lyrics).
"""

from jukebox_infer import Jukebox

# Initialize model (checkpoints auto-download on first use)
model = Jukebox(model_name="1b_lyrics", device="cuda")

# Load model (downloads checkpoints automatically if needed)
# Minimum duration for 1b_lyrics is ~18 seconds
model.load(sample_length_in_seconds=20, n_samples=1)

# Generate music
print("Generating music... (this will take a while - ~1-2 min per second of audio)")
audio = model.generate(
    artist="The Beatles",
    genre="Rock",
    duration_seconds=20,
    output_path="output.wav"
)

print(f"✓ Generated audio saved to output.wav")
print(f"  Shape: {audio.shape}")
print("Done!")
