FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# -----------------------------
# Install python + pip
# -----------------------------
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev curl git wget && \
    rm -rf /var/lib/apt/lists/*

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# make it available in PATH
ENV PATH="/root/.local/bin:$PATH"

# -----------------------------
# Python packages
# -----------------------------
RUN pip3 install --upgrade pip \
    && pip3 install "numpy<2" "opencv-python-headless"

# -----------------------------
# Install PyTorch 1.12.1 + CUDA 11.3
# -----------------------------
RUN pip3 install --no-cache-dir --default-timeout=1000 --progress-bar on \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113 

# Command to build:
# docker build --no-cache -t sparse_planner .

# Command to run the first time:
# docker run -it --name sparse_planner_container -v /mnt:/mnt -v /home/luce_gi:/home/luce_gi --network host --shm-size=30g --gpus all sparse_planner bash
