#!/bin/bash

# Update package list and install system dependencies
apt-get update && apt-get install -y libgl1-mesa-glx libgl1-mesa-dri python3-venv python3-pip cmake python3-distutils

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and setuptools to the latest version
pip install --upgrade pip setuptools

# Install build dependencies using a pyproject.toml
echo "[build-system]
requires = ['setuptools', 'wheel']
build-backend = 'setuptools.build_meta'" > pyproject.toml

# Install Python packages from requirements.txt
pip install -r requirements.txt

# Create the Streamlit configuration directory
mkdir -p ~/.streamlit/

# Create the Streamlit configuration file
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
