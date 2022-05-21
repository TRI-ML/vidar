FROM nvidia/cuda:11.3.1-devel-ubuntu18.04

ENV PROJECT=vidar
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHON_VERSION=3.8
ENV PYTORCH_VERSION=1.10.0+cu113
ENV TORCHVISION_VERSION=0.11.1+cu113
ENV CUDNN_VERSION=8.2.1.32-1+cuda11.3
ENV NCCL_VERSION=2.9.9-1+cuda11.3

# Install basic libraries
RUN apt-get update && apt-get install -y \
    build-essential cmake g++-4.8 git curl docker.io vim wget ca-certificates

# Install python and pip
RUN apt-get install -y python${PYTHON_VERSION} python3-pip 
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip 

# Install pytorch and torchvision
RUN pip install \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install CUDNN and NCCL
RUN apt-get install -y \
    libcudnn8=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION}

# Install extra packages (apt-get)
RUN apt-get install -y \
    ffmpeg \
    tmux

# Install extra packages (pip)
RUN pip install \
    tqdm==4.61.0 \
    boto3==1.17.83 \
    termcolor==1.1.0 \
    pyyaml==5.4.1 \
    wandb==0.10.31 \
    opencv-python==4.5.2.52 \
    flow_vis==0.1 \
    matplotlib==3.3.4 \
    fire==0.4.0 \
    pyquaternion==0.9.9 \
    pandas==1.1.5 \
    xarray==0.16.2 \
    diskcache==5.2.1 \
    tenacity==7.0.0 \
    pycocotools==2.0.2 \
    awscli==1.19.101 \
#    timm==0.4.9 \
    ref==0.0.2.2 \
    positional-encodings==4.0.0 \
    einops==0.3.2 \
    wget \
    ftfy \
    regex \
    tqdm

# Install CamViz dependencies
RUN pip install \
    pygame==2.0.1 \
    PyOpenGL==3.1.5 \
    PyOpenGL-accelerate==3.1.5
RUN apt-get install -y \
    mesa-utils \
    freeglut3-dev \
    libsdl2-2.0-0 \
    python-pygame

# Install PyTorch3D
RUN pip install pytorch3d

# Install CuPY
RUN pip install cupy

# Install extras (should be moved to top when stable)
RUN pip install lpips wget scikit-image pyhocon dotmap path sacremoses filelock huggingface_hub
RUN pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# Install DGP (dataset utils)
WORKDIR /workspace
RUN git clone https://github.com/VitorGuizilini-TRI/dgp.git
ENV PYTHONPATH="/workspace/dgp:$PYTHONPATH"

# Create workspace folder
RUN mkdir -p /workspace/experiments
RUN mkdir -p /workspace/${PROJECT}
WORKDIR /workspace/${PROJECT}
# Copy project to workspace folder
COPY . /workspace/${PROJECT}

# Set environment variables
ENV PYTHONPATH="/workspace/${PROJECT}:$PYTHONPATH"
ENV PYTHONPATH="/workspace/${PROJECT}/externals/camviz:$PYTHONPATH"
