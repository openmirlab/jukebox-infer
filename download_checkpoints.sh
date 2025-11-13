#!/bin/bash

# Download 1b_lyrics model checkpoints
# NOTE: Upsamplers are shared with 5b model (from hparams.py analysis)

echo "Downloading 1b_lyrics model checkpoints..."
echo "Note: VQ-VAE and upsamplers are shared with 5b model"
echo ""

# Create directories
mkdir -p "$HOME/.cache/jukebox/models/5b"
mkdir -p "$HOME/.cache/jukebox/models/1b_lyrics"

# VQ-VAE (7.4MB) - shared across all models
if [ ! -f "$HOME/.cache/jukebox/models/5b/vqvae.pth.tar" ]; then
    echo "[1/4] Downloading VQ-VAE (7.4MB)..."
    wget -c -O "$HOME/.cache/jukebox/models/5b/vqvae.pth.tar" \
        https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar
else
    echo "[1/4] VQ-VAE already downloaded ✓"
fi

# Upsampler level 0 (~2.2GB) - shared from 5b model
if [ ! -f "$HOME/.cache/jukebox/models/5b/prior_level_0.pth.tar" ]; then
    echo "[2/4] Downloading upsampler_level_0 (2.2GB, from 5b)..."
    wget -c -O "$HOME/.cache/jukebox/models/5b/prior_level_0.pth.tar" \
        https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_0.pth.tar
else
    echo "[2/4] Upsampler level 0 already downloaded ✓"
fi

# Upsampler level 1 (~2.2GB) - shared from 5b model
if [ ! -f "$HOME/.cache/jukebox/models/5b/prior_level_1.pth.tar" ]; then
    echo "[3/4] Downloading upsampler_level_1 (2.2GB, from 5b)..."
    wget -c -O "$HOME/.cache/jukebox/models/5b/prior_level_1.pth.tar" \
        https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_1.pth.tar
else
    echo "[3/4] Upsampler level 1 already downloaded ✓"
fi

# Top level prior (~1.8GB) - 1b_lyrics specific
if [ ! -f "$HOME/.cache/jukebox/models/1b_lyrics/prior_level_2.pth.tar" ]; then
    echo "[4/4] Downloading prior_1b_lyrics (1.8GB)..."
    wget -c -O "$HOME/.cache/jukebox/models/1b_lyrics/prior_level_2.pth.tar" \
        https://openaipublic.azureedge.net/jukebox/models/1b_lyrics/prior_level_2.pth.tar
else
    echo "[4/4] Prior 1b_lyrics already downloaded ✓"
fi

echo ""
echo "✓ Download complete! Total size: ~6.2GB"
echo ""
echo "Files in 5b (shared):"
ls -lh "$HOME/.cache/jukebox/models/5b/"
echo ""
echo "Files in 1b_lyrics (specific):"
ls -lh "$HOME/.cache/jukebox/models/1b_lyrics/"
