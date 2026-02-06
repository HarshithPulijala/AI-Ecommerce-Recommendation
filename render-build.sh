#!/usr/bin/env bash
# Render build script with Git LFS support

set -e  # Exit on error

echo "=== Starting Render Build ==="
echo "Node environment: $(node --version 2>/dev/null || echo 'N/A')"
echo "Python version: $(python --version)"
echo "Git version: $(git --version)"

# Try to install Git LFS
echo "=== Installing Git LFS ==="
if command -v git-lfs &> /dev/null; then
    echo "Git LFS already installed"
else
    echo "Installing Git LFS from apt..."
    apt-get update -qq
    apt-get install -y git-lfs 2>&1 | tail -5 || echo "apt install failed, trying alternative method"
fi

# Initialize and pull LFS files
echo "=== Pulling Git LFS files ==="
git lfs install --local
git lfs pull

echo "=== Verifying data files ==="
ls -lah data/processed/ | head -10
ls -lah models/ | head -10

# Check if files were actually downloaded (not just pointers)
echo "Checking file types..."
file data/processed/interactions_clean.csv | head -c 100 || echo "Could not check file type"
file models/svd_model.pkl | head -c 100 || echo "Could not check file type"

echo "=== Installing Python dependencies ==="
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "=== Build complete ==="
