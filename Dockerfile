# Build the image and tag it for easier later reference
# Example:
#   docker build -t reekfsort .

# Base image: Nvidia PyTorch https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Update and install necessary packages
RUN apt update && apt install -y git

# Set the parent working directory
WORKDIR /usr/src

# Copy the local working tree into the image.
COPY . boxmot

# Set the working directory to the cloned repository
WORKDIR /usr/src/boxmot

# Install uv and sync dependencies
RUN pip install uv && \
    uv sync --all-extras --all-groups

# ------------------------------------------------------------------------------

# A Docker container exits when its main process finishes, which in this case is bash.
# To avoid this, use detach mode.

# Run interactively with all GPUs accessible:
#   docker run -it --gpus all reekfsort bash

# Run interactively with specific GPUs accessible (e.g., first and third GPU):
#   docker run -it --gpus '"device=0,2"' reekfsort bash

# Run in detached mode (if you exit the container, it won't stop):
# Create a detached Docker container from an image:
#   docker run -it --gpus all -d reekfsort

# Access the running container:
#   docker exec -it <container_id> bash

# When you are done with the container, stop it by:
#   docker stop <container_id>
