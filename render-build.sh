#!/usr/bin/env bash
# Render build script with Git LFS support

echo "=== Installing Git LFS ==="
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs

echo "=== Initializing Git LFS ==="
git lfs install

echo "=== Pulling LFS files ==="
git lfs pull

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Build complete ==="
