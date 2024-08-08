#!/bin/bash

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

# Install system dependencies
apt-get update && apt-get install -y python3-distutils libgl1-mesa-glx libgl1-mesa-dri

# Update pip
pip install --upgrade pip
