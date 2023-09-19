FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV PROJECT=vidar
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTORCH_VERSION=1.12.1+cu113
ENV TORCHVISION_VERSION=0.13.1+cu113
ENV TORCHAUDIO_VERSION=0.12.1

ENV CUDNN_VERSION=8.2.1.32-1+cuda11.3
ENV NCCL_VERSION=2.9.9-1+cuda11.3
ENV DEBIAN_FRONTEND=noninteractive

# Install basic libraries
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl docker.io vim wget ca-certificates

# Install python and pip
RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# Install pytorch
RUN pip install  \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install CUDNN and NCCL
RUN apt-get install -y \
    libcudnn8=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION}

# Install extra packages (apt-get)
RUN apt-get install -y \
    ffmpeg \
    tmux

# Install extra packages (pip)
RUN pip install                 \
    tqdm==4.61.0                \
    boto3==1.17.83              \
    termcolor==1.1.0            \
    pyyaml==5.4.1               \
    wandb==0.10.31              \
    opencv-python==4.5.2.52     \
    flow_vis==0.1               \
    matplotlib==3.3.4           \
    fire==0.4.0                 \
    pyquaternion==0.9.9         \
    pandas==1.1.5               \
    xarray==0.16.2              \
    diskcache==5.2.1            \
    tenacity==7.0.0             \
    awscli==1.19.101            \
    ref==0.0.2.2                \
    positional-encodings==4.0.0 \
    einops==0.3.2               \
    wget==3.2                   \
    kornia==0.6.8               \
    lpips==0.1.4                \
    scikit-image==0.19.3        \
    timm==0.6.11

# Install CamViz dependencies
RUN pip install                 \
    pygame==2.0.1               \
    PyOpenGL==3.1.5             \
    PyOpenGL-accelerate==3.1.5
RUN apt-get install -y          \
    mesa-utils                  \
    freeglut3-dev               \
    libsdl2-2.0-0               \
    python-pygame

# Install CLIP
RUN pip install ftfy regex
RUN pip install git+https://github.com/openai/CLIP.git
# Install torch-scatter
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
# Install PyTorch3D
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# Install LieTorch
RUN pip install git+https://github.com/princeton-vl/lietorch.git@91887c3be4fe72aefe4af1deb67f2ea9ce17a049
# Install nuScenes devkit
RUN pip install nuscenes-devkit==1.1.7
# Upgrade protobuf
RUN pip install --upgrade protobuf==3.20.0
# Install transformer stuff
RUN pip install sacremoses huggingface_hub transformers
# Install open clip
RUN pip install open_clip_torch
# Install more transformers stuff
RUN pip install mmcv taming-transformers taming-transformers-rom1504
# Install seaborn for extrinsic error map visualization
RUN pip install seaborn==0.12.2

# Expose jupyter notebook port
EXPOSE 8888
ENV PROJECT=vidar_release

# Create workspace folder
RUN mkdir -p /workspace/experiments
RUN mkdir -p /workspace/${PROJECT}

# Copy project to workspace folder
WORKDIR /workspace/${PROJECT}
COPY . /workspace/${PROJECT}

# Set environment variables
ENV PYTHONPATH="/workspace/${PROJECT}:$PYTHONPATH"
ENV PYTHONPATH="/workspace/${PROJECT}/externals/camviz:$PYTHONPATH"
ENV PYTHONPATH="/workspace/${PROJECT}/externals/efm_datasets:$PYTHONPATH"
