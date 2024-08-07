# setup.sh
#!/bin/bash

# Install setuptools and distutils using apt
apt-get update && apt-get install -y python3-setuptools python3-distutils

# Install setuptools and distutils using pip
pip install setuptools

# Install Python packages
pip install -r requirements.txt
