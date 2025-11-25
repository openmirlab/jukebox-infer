# Publishing to PyPI

This document explains how to publish jukebox-infer to PyPI using GitHub Actions and trusted publishing.

---

## Overview

Jukebox-infer uses **GitHub Actions** with **PyPI Trusted Publishing** for automated package releases. This eliminates the need for manual API tokens and provides a secure, auditable publishing process.

---

## Prerequisites

### 1. Configure PyPI Trusted Publishing

Before publishing, you must configure trusted publishing on PyPI:

1. Go to [PyPI](https://pypi.org/) and log in
2. Navigate to your account settings
3. Go to "Publishing" → "Add a new publisher"
4. Fill in the form:
   - **PyPI Project Name**: `jukebox-infer`
   - **Owner**: `openmirlab`
   - **Repository name**: `jukebox-infer`
   - **Workflow name**: `publish.yml`
   - **Environment name**: (leave empty or use `pypi`)
5. Save the publisher configuration

**Note**: This only needs to be done once. After the first release, PyPI will automatically trust subsequent releases from the same GitHub workflow.

---

## Publishing a New Version

### Step 1: Update Version Number

Edit `pyproject.toml` and update the version number:

```toml
[project]
name = "jukebox-infer"
version = "0.1.1"  # ← Update this
```

Follow [semantic versioning](https://semver.org/):
- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backwards compatible
- **Patch** (0.1.1): Bug fixes, backwards compatible

### Step 2: Commit and Push Changes

```bash
git add pyproject.toml
git commit -m "Bump version to 0.1.1"
git push origin main
```

### Step 3: Create a GitHub Release

1. Go to the [GitHub Releases page](https://github.com/openmirlab/jukebox-infer/releases)
2. Click "Draft a new release"
3. Click "Choose a tag" and create a new tag matching the version:
   - Tag: `v0.1.1`
   - Target: `main`
4. Fill in the release details:
   - **Release title**: `v0.1.1` (or descriptive title)
   - **Description**: Changelog and release notes
5. Click "Publish release"

### Step 4: Monitor the Workflow

The GitHub Actions workflow will automatically:
1. Checkout the code
2. Set up Python 3.10
3. Install build dependencies
4. Build the package (wheel + source dist)
5. Validate the package with `twine check`
6. Publish to PyPI using trusted publishing

**Monitor the workflow:**
- Go to [Actions](https://github.com/openmirlab/jukebox-infer/actions)
- Find the "Publish to PyPI" workflow run
- Check for any errors

### Step 5: Verify Publication

After the workflow completes successfully:

1. Check PyPI: https://pypi.org/project/jukebox-infer/
2. Verify the new version is listed
3. Test installation:
   ```bash
   pip install --upgrade jukebox-infer
   pip show jukebox-infer  # Should show new version
   ```

---

## Manual Publishing (Emergency Only)

If GitHub Actions is unavailable, you can publish manually:

### Prerequisites

```bash
pip install build twine
```

### Build and Publish

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build

# Check the package
twine check dist/*

# Upload to PyPI (requires API token)
twine upload dist/*
```

**Note**: This requires a PyPI API token. The automated workflow is strongly preferred.

---

## Workflow Configuration

The publishing workflow is defined in `.github/workflows/publish.yml`:

```yaml
on:
  release:
    types: [published]  # Triggers on GitHub Release
  workflow_dispatch:      # Manual trigger option
```

### Workflow Permissions

```yaml
permissions:
  id-token: write  # Required for trusted publishing (OIDC)
  contents: read   # Required to checkout code
```

### Build Process

1. **Build**: Uses `python -m build` to create wheel and source distribution
2. **Validate**: Uses `twine check` to validate package metadata
3. **Publish**: Uses `pypa/gh-action-pypi-publish@release/v1` for secure OIDC authentication

---

## Troubleshooting

### Error: "Publisher not configured"

**Solution**: Configure PyPI trusted publishing (see Prerequisites above)

### Error: "Version already exists"

**Solution**: Update the version number in `pyproject.toml` before creating a release

### Error: "Invalid package metadata"

**Solution**: Run `twine check dist/*` locally to identify metadata issues

### Workflow Fails with Permission Error

**Solution**: Ensure the workflow has `id-token: write` permission in the workflow file

---

## Release Checklist

Before creating a release:

- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG (if applicable)
- [ ] Run tests locally: `pytest`
- [ ] Build locally: `python -m build`
- [ ] Check package: `twine check dist/*`
- [ ] Commit and push version bump
- [ ] Create GitHub Release with matching tag
- [ ] Monitor GitHub Actions workflow
- [ ] Verify package on PyPI
- [ ] Test installation: `pip install jukebox-infer`

---

## Additional Resources

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Packaging User Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)

---

**Last Updated**: 2025-01-25
