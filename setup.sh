#!/bin/bash

# Check if Docker is installed
if ! [ -x "$(command -v docker)" ]; then
  echo 'Error: Docker is not installed.' >&2
  exit 1
fi

# Build the Docker image
docker build -t streamlit-app .

# Run the Docker container
docker run -p 8501:8501 streamlit-app
