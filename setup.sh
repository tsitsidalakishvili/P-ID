# setup.sh
#!/bin/bash

# Install setuptools using apt
apt-get update && apt-get install -y python3-setuptools

# Install setuptools using pip
pip install setuptools

# Install Python packages
pip install -r requirements.txt
