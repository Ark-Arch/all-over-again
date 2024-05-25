#!/bin/bash

# Setup Python environment
python3 -m venv venv
source venv/bin/activate

# Install system dependencies
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx

# Install Python packages
pip install tensorflow
pip install tensorflow_datasets
pip install scikit-image
pip install mltu
pip install stow
pip install tf2onnx
pip install pydot graphviz
pip install scikit-learn

# Clear terminal and show a message
clear
echo "YOUR WORKSPACE IS READY FOR CODING!"
