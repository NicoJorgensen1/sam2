#!/bin/bash

# Ensure SAM2DIR is set or exit
if [ -z "$SAM2DIR" ]; then
    echo "SAM2DIR is not set. Exiting."
    exit 1
fi

# Navigate to SAM2DIR
cd "$SAM2DIR" || { echo "Failed to change directory to SAM2DIR: $SAM2DIR"; exit 1; }

# Login to ecr 
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 251553394525.dkr.ecr.eu-west-1.amazonaws.com

# Build the docker image
docker build -f .devcontainer/regular_SAM2/Dockerfile -t sam2_image .

# Tag the docker image with the ecr repo and the 'latest' tag
docker tag sam2_image:latest 251553394525.dkr.ecr.eu-west-1.amazonaws.com/ai_base_images_for_development:sam2_image

# Push the docker image to ecr 
docker push 251553394525.dkr.ecr.eu-west-1.amazonaws.com/ai_base_images_for_development:sam2_image
