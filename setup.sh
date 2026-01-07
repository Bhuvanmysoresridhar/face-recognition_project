#!/bin/bash

# Face Recognition Project Setup Script
# This script helps install all dependencies including dlib

echo "🔧 Setting up Face Recognition Project..."

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv ../venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source ../venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $PYTHON_VERSION"

# Install cmake first (needed for dlib compilation)
echo "🔨 Installing cmake (required for dlib)..."
pip install cmake

# Check if cmake is available
if command -v cmake &> /dev/null; then
    echo "✅ cmake found in system PATH"
else
    echo "⚠️  cmake not in PATH, but pip package should work"
fi

# Install requirements
echo "📥 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "import face_recognition; import cv2; import numpy; print('✅ All packages installed successfully!')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete! You can now run:"
    echo "   python main.py"
else
    echo ""
    echo "❌ Installation failed. Please check the error messages above."
    echo "   Try installing cmake via Homebrew: brew install cmake"
    exit 1
fi

