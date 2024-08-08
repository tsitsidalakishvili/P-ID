#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y libgl1-mesa-glx libgl1-mesa-dri

# Install Python distutils if not already present
apt-get install -y python3-setuptools

# Upgrade setuptools and pip to make sure we have the latest versions
pip install --upgrade setuptools pip

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
