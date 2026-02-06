#!/usr/bin/env bash
# Render build script with Git LFS support

echo "=== Starting Render Build ==="
echo "Python version: $(python --version)"
echo "Git version: $(git --version)"

# Install Git LFS - continue even if this fails
echo "=== Installing Git LFS ==="
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS not found, attempting installation..."
    apt-get update -qq 2>/dev/null || true
    apt-get install -y git-lfs 2>/dev/null || echo "Warning: Git LFS installation via apt failed"
fi

# Initialize LFS and pull files
echo "=== Pulling Git LFS files ==="
git lfs install --local 2>/dev/null || true
git lfs pull || echo "Warning: git lfs pull failed, files may not be available"

# Verify data files exist
echo "=== Verifying files ==="
if [ -f "data/processed/interactions_clean.csv" ]; then
    echo "✓ interactions_clean.csv found"
else
    echo "✗ interactions_clean.csv NOT found"
fi

if [ -f "models/svd_model.pkl" ]; then
    echo "✓ svd_model.pkl found"
else
    echo "✗ svd_model.pkl NOT found"
fi

# Install Python dependencies - this must succeed
echo "=== Installing Python dependencies ==="
pip install --upgrade pip -q || exit 1
pip install -r requirements.txt -q || exit 1

echo "=== Build complete ==="

