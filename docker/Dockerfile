FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    git-lfs \
    graphviz-dev \
    wget \
    unzip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Miniconda
ENV CONDA_DIR=/opt/conda
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Clone and set up the project
RUN git lfs install && \
    mkdir tmp && \
    wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth?download=true -O tmp/control_sd15_canny.pth && \
    wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth?download=true -O tmp/control_v11p_sd15_canny.pth && \
    mv tmp/control_sd15_canny.pth easi-tex/models/ControlNet/models && \
    mv tmp/control_v11p_sd15_canny.pth easi-tex/models/ControlNet/models && \
    git clone https://huggingface.co/h94/IP-Adapter tmp/ip_adapter && \
    mv tmp/ip_adapter/models/image_encoder easi-tex/ip_adapter && \
    mv tmp/ip_adapter/models/ip-adapter-plus_sd15.bin easi-tex/ip_adapter && \
    wget https://aspis.cmpt.sfu.ca/projects/singapo/data/pm.zip -O tmp/pm.zip && \
    export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE && \
    unzip tmp/pm.zip -d data && \
    wget https://aspis.cmpt.sfu.ca/projects/singapo/ckpts/singapo_ckpt.zip -O tmp/singapo_ckpt.zip && \
    mkdir -p singapo/exps/singapo && \
    unzip tmp/singapo_ckpt.zip -d singapo/exps/singapo && \
    rm -rf tmp

# Create conda environment
COPY environment.yml /tmp/environment.yml
COPY requirements.txt /tmp/requirements.txt
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

RUN git clone https://github.com/EgorFokin/texture_singapo


# Set working directory
WORKDIR /texture_singapo

# Default command
CMD [ "bash" ]
