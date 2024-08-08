#!/bin/bash

# Update package list and install system dependencies
apt-get update && apt-get install -y libgl1-mesa-glx libgl1-mesa-dri python3-venv python3-pip cmake

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Ensure pip and setuptools are up to date
pip install --upgrade pip setuptools

# Explicitly install distutils
pip install setuptools==58.0.4

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
