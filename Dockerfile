# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Install system dependencies and Python
RUN apt update && \
    apt install -y \
        python3.10 \
        python3.10-dev \
        python3-pip \
        build-essential \
        cmake \
        libpng-dev \
        pkg-config \
        git \
        wget \
        unzip \
        && apt clean && \
        rm -rf /var/lib/apt/lists/*

# Create symbolic links for python3.10 if they don't exist
RUN if [ ! -L /usr/bin/python3 ]; then \
        ln -s /usr/bin/python3.10 /usr/bin/python3; \
    fi && \
    if [ ! -L /usr/bin/python ]; then \
        ln -s /usr/bin/python3.10 /usr/bin/python; \
    fi

# Upgrade pip and install basic Python tools
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /opt

# Create application directories
RUN mkdir -p /opt/src /opt/data

# Download and extract template files
RUN cd /opt/data && \
    # Download and extract OASIS-30 Atropos template
    wget -O OASIS-30_Atropos_template.zip "https://osf.io/rh9km/?action=download&version=1" && \
    unzip -o OASIS-30_Atropos_template.zip && \
    rm OASIS-30_Atropos_template.zip && \
    # Download and extract OASIS-TRT-20 brains
    wget -O OASIS-TRT-20_brains.zip "https://files.osf.io/v1/resources/hs8am/providers/osfstorage/57c1a8f06c613b01f98d68a9/?zip=" && \
    unzip -o OASIS-TRT-20_brains.zip -d OASIS-TRT-20_brains && \
    rm OASIS-TRT-20_brains.zip && \
    # Download and extract OASIS-TRT-20 DKT31 CMA labels
    wget -O OASIS-TRT-20_DKT31_CMA_labels_v2.zip "https://files.osf.io/v1/resources/hs8am/providers/osfstorage/57c1a8ffb83f690201c4a8be/?zip=" && \
    unzip -o OASIS-TRT-20_DKT31_CMA_labels_v2.zip -d OASIS-TRT-20_DKT31_CMA_labels_v2 && \
    rm OASIS-TRT-20_DKT31_CMA_labels_v2.zip && \
    # Download joint fusion labels
    wget -O OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz "https://osf.io/download/nxg5t/"

# Copy the application code
COPY src/ /opt/src/
COPY requirements.txt /opt/requirements.txt

# Install Python dependencies
RUN cd /opt && \
    python -m pip install -r requirements.txt && \
    # Install ants_seg_to_nidm
    cd /opt/src/ants_seg_to_nidm && \
    python -m pip install -e . && \
    # Install specific versions of rdflib packages
    python -m pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/opt:$PYTHONPATH
ENV PATH=/usr/local/bin:$PATH

# Set the entrypoint
ENTRYPOINT ["python", "/opt/src/run.py"]