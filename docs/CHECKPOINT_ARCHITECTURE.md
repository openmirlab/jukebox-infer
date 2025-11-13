# Jukebox Checkpoint Architecture

**Date:** 2025-10-03
**Discovered by:** Analyzing original jukebox/hparams.py

---

## Model Structure

All Jukebox models (5b, 5b_lyrics, 1b_lyrics) share the same foundational components:

### Shared Components (from 5b model)

```
~/.cache/jukebox/models/5b/
├── vqvae.pth.tar          (7.4MB)   ✓ Encoder/decoder
├── prior_level_0.pth.tar  (2.2GB)   ✓ Upsampler level 0
└── prior_level_1.pth.tar  (2.2GB)   ✓ Upsampler level 1
```

**These are shared across ALL models!**

### Model-Specific Components

Only the **top-level prior (level 2)** is model-specific:

```
~/.cache/jukebox/models/5b/
└── prior_level_2.pth.tar  (2.2GB)   [5b model]

~/.cache/jukebox/models/5b_lyrics/
└── prior_level_2.pth.tar  (2.2GB)   [5b_lyrics model]

~/.cache/jukebox/models/1b_lyrics/
└── prior_level_2.pth.tar  (1.8GB)   [1b_lyrics model]
```

---

## URL Structure

From `jukebox/hparams.py`:

```python
REMOTE_PREFIX = 'https://openaipublic.azureedge.net/'

# Shared components (defined in vqvae hparams)
restore_vqvae = REMOTE_PREFIX + 'jukebox/models/5b/vqvae.pth.tar'

# Shared components (defined in upsampler_level_0 hparams)
restore_prior = REMOTE_PREFIX + 'jukebox/models/5b/prior_level_0.pth.tar'

# Shared components (defined in upsampler_level_1 hparams)
restore_prior = REMOTE_PREFIX + 'jukebox/models/5b/prior_level_1.pth.tar'

# Model-specific (defined in prior_1b_lyrics hparams)
restore_prior = REMOTE_PREFIX + 'jukebox/models/1b_lyrics/prior_level_2.pth.tar'
```

---

## Download Requirements

### For 1b_lyrics Model

```bash
# Total: ~6.2GB

# Shared (5b directory)
https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar           # 7.4MB
https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_0.pth.tar  # 2.2GB
https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_1.pth.tar  # 2.2GB

# Model-specific (1b_lyrics directory)
https://openaipublic.azureedge.net/jukebox/models/1b_lyrics/prior_level_2.pth.tar  # 1.8GB
```

### For 5b_lyrics Model

```bash
# Total: ~8.4GB

# Shared (5b directory)
https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar           # 7.4MB
https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_0.pth.tar  # 2.2GB
https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_1.pth.tar  # 2.2GB

# Model-specific (5b_lyrics directory)
https://openaipublic.azureedge.net/jukebox/models/5b_lyrics/prior_level_2.pth.tar  # 2.2GB
```

---

## Why This Matters

**Common Mistake:** Trying to download `jukebox/models/1b_lyrics/prior_level_0.pth.tar`

**Result:** 404 Error - file doesn't exist!

**Correct Approach:** Download from `jukebox/models/5b/prior_level_0.pth.tar`

---

## Implementation in jukebox-infer

Checkpoints are automatically downloaded when you first use a model. The download system:

1. Checks if checkpoints exist in `~/.cache/jukebox/models/`
2. Downloads missing checkpoints automatically with progress bars
3. Caches checkpoints for future use
4. Supports manual pre-download via `download_checkpoints()` function or `download_checkpoints.sh` script

---

## Model Hierarchy

```
                    VQ-VAE (shared)
                        │
            ┌───────────┼───────────┐
            │           │           │
     Upsampler L0  Upsampler L1    │
      (shared)      (shared)        │
            │           │           │
            └───────────┼───────────┘
                        │
                   Prior L2
              (model-specific)
                        │
                ┌───────┼───────┐
                │       │       │
              5b    5b_lyrics  1b_lyrics
```

**Bottom line:** All models use the same VQ-VAE and upsamplers. Only the top-level prior differs.

---

## References

- Azure CDN: `https://openaipublic.azureedge.net/jukebox/`
- OpenAI Jukebox: https://github.com/openai/jukebox
