# Installation Guide for Face Recognition Project

## Problem
`dlib` (required by `face-recognition`) fails to build on macOS because it requires C++ compilation tools.

## Solution Options

### Option 1: Install cmake via Homebrew (Recommended)
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install cmake
brew install cmake

# Activate your virtual environment
source ../venv/bin/activate

# Now install requirements
pip install -r requirements.txt
```

### Option 2: Install cmake via pip (Easier)
```bash
# Activate your virtual environment
source ../venv/bin/activate

# Install cmake first
pip install cmake

# Then install requirements
pip install -r requirements.txt
```

### Option 3: Use conda (If you have Anaconda/Miniconda)
```bash
# Create a new conda environment
conda create -n face_recognition python=3.11 -y
conda activate face_recognition

# Install dlib from conda-forge (pre-built, no compilation needed)
conda install -c conda-forge dlib -y

# Install other requirements
pip install opencv-python==4.8.1.78 face-recognition==1.3.0 numpy==1.24.3
```

### Option 4: Install Xcode Command Line Tools (If missing)
```bash
xcode-select --install
```

Then try Option 1 or 2 again.

## Quick Start (After Installation)
```bash
# Activate virtual environment
source ../venv/bin/activate

# Run the face recognition system
python main.py
```

