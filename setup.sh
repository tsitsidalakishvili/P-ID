# setup.sh
#!/bin/bash

# Install setuptools
apt-get update && apt-get install -y python3-setuptools

# Install Python packages
pip install -r requirements.txt
