# Development Principles

## Package Management

### Use UV for Environment Management

This project uses [uv](https://github.com/astral-sh/uv) as the package and virtual environment manager.

**Key Rules:**

1. **Virtual Environment Management**: Always use `uv` to manage the virtual environment
   ```bash
   # Install dependencies
   uv sync

   # Add new dependencies
   uv add <package-name>
   ```

2. **Running Python Scripts**: Always use `uv run` instead of `python` directly
   ```bash
   # ✅ Correct
   uv run python quick_infer.py
   uv run python examples/basic_generation.py

   # ❌ Incorrect
   python quick_infer.py
   ```

## Version Compatibility

### Critical Packages Reference

The following packages are critical for compatibility and should use these recommended versions:

- **PyTorch Ecosystem**:
  - `torch>=2.7.0` (preferably `torch==2.7.1`)
  - `torchvision>=0.22.0` (preferably `torchvision==0.22.1`)
  - `torchaudio>=2.7.0` (preferably `torchaudio==2.7.1`)

- **CUDA Dependencies**:
  - `nvidia-cuda-runtime-cu12>=12.6.77`
  - `nvidia-cudnn-cu12>=9.5.1.17`

- **Audio Processing**:
  - `librosa>=0.11.0`
  - `soundfile>=0.13.0`

### Before Adding Dependencies

When adding new dependencies, especially those related to ML/DL frameworks:

1. **Check existing versions**: Review `pyproject.toml` to see if the package is already specified
2. **Match compatible versions**: Use compatible version constraints to avoid conflicts
3. **Test integration**: After adding, verify the package installs correctly and works with existing dependencies

Example workflow:
```bash
# 1. Check existing version in pyproject.toml
grep "torch" pyproject.toml

# 2. Add with compatible version
uv add "torch>=2.7.0"

# 3. Verify no conflicts
uv sync
```

## Code Style

### Formatting

- **Ruff**: Use `ruff` for linting and import sorting
- **Black**: Use `black` for code formatting (line length: 100)
- **Type Hints**: Add type hints where appropriate (mypy configured but not strict)

### Running Linters

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Type checking (optional)
uv run mypy jukebox_infer
```

## Testing

### Test Structure

- Tests should be in the `tests/` directory
- Use pytest for testing
- Test files should follow the pattern `test_*.py`

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=jukebox_infer --cov-report=html
```

## Project-Specific Guidelines

### Respecting Original Jukebox

- **Attribution**: Always maintain proper attribution to OpenAI Jukebox
- **License**: Keep MIT license and include attribution in LICENSE file
- **Citation**: Include citation information in README and documentation
- **Code Comments**: When modifying original code, add comments explaining changes

### Inference-Only Focus

- **No Training Code**: This is an inference-only package - do not add training functionality
- **Minimal Dependencies**: Keep dependencies minimal - avoid adding heavy training-related packages
- **Single GPU**: Optimize for single-GPU inference, not distributed training

### API Design

- **High-Level API**: The `Jukebox` class in `api.py` should remain simple and user-friendly
- **Low-Level Access**: Keep low-level functions available for advanced users
- **Backward Compatibility**: Maintain backward compatibility when possible

## Rationale

### Why UV?

- **Fast**: UV is significantly faster than pip for dependency resolution
- **Consistent**: Lockfile ensures reproducible builds across environments
- **Modern**: Better dependency resolution algorithm

### Why Version Compatibility Matters?

PyTorch and CUDA libraries are notorious for version incompatibilities. By maintaining compatible versions:

- **Prevents runtime errors** from incompatible binary dependencies
- **Reduces installation time** by avoiding duplicate package versions
- **Ensures consistent behavior** across different environments
- **Simplifies debugging** when issues arise

### Common Conflict Scenarios to Avoid

- Different PyTorch versions requiring different CUDA toolkit versions
- Audio processing libraries (torchaudio, librosa) with conflicting backend requirements
- NumPy version conflicts with PyTorch

## Quick Reference

```bash
# Setup project
uv sync

# Run quick inference script
uv run python quick_infer.py

# Run examples
uv run python examples/basic_generation.py

# Format and lint
uv run ruff format . && uv run ruff check .

# Add dependency
uv add <package-name>

# Update dependencies
uv lock --upgrade

# Check existing version in pyproject.toml
grep "<package-name>" pyproject.toml
```
