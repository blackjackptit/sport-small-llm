#!/bin/bash
# Setup script for Sports Domain LLM

set -e

echo "=================================="
echo "Sports Domain LLM Setup"
echo "=================================="

# Create virtual environment
echo -e "\n[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "\n[2/4] Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for local testing, GPU version for training)
echo -e "\n[3/4] Installing PyTorch..."
# For Mac (MPS support)
pip install torch torchvision torchaudio

# Install other dependencies
echo -e "\n[4/4] Installing dependencies..."
pip install -r requirements.txt

echo -e "\n=================================="
echo "Setup complete!"
echo "=================================="
echo -e "\nTo activate the environment:"
echo "  source venv/bin/activate"
echo -e "\nTo run local test:"
echo "  python scripts/run_local_test.py"
echo -e "\nTo prepare data for SageMaker:"
echo "  python sagemaker/prepare_data.py --help"
