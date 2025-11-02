#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit

echo "Starting Lexi-Guide build process..."

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Use simple requirements to avoid compilation
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Build completed successfully!"
