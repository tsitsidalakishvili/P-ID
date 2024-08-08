#!/bin/bash

# Update package list and install system dependencies
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    python3.11-venv \
    python3.11-distutils

# Create and activate a virtual environment
python3.11 -m venv ~/.venv
source ~/.venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Install the required packages from requirements.txt
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
