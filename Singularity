Bootstrap: library
From: ubuntu:22.04

%post
    # Update and install basic system dependencies
    apt update && \
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
    if [ ! -L /usr/bin/python3 ]; then
        ln -s /usr/bin/python3.10 /usr/bin/python3
    fi
    
    if [ ! -L /usr/bin/python ]; then
        ln -s /usr/bin/python3.10 /usr/bin/python
    fi

    # Upgrade pip and install basic Python tools
    python3 -m pip install --upgrade pip setuptools wheel

    # Create application directories
    mkdir -p /opt/src
    mkdir -p /opt/data

    # Download and extract template files
    cd /opt/data

    # Download and extract OASIS-30 Atropos template
    wget -O OASIS-30_Atropos_template.zip "https://osf.io/rh9km/?action=download&version=1"
    unzip -o OASIS-30_Atropos_template.zip
    rm OASIS-30_Atropos_template.zip

    # Download and extract OASIS-TRT-20 brains
    wget -O OASIS-TRT-20_brains.zip "https://files.osf.io/v1/resources/hs8am/providers/osfstorage/57c1a8f06c613b01f98d68a9/?zip="
    unzip -o OASIS-TRT-20_brains.zip -d OASIS-TRT-20_brains
    rm OASIS-TRT-20_brains.zip

    # Download and extract OASIS-TRT-20 DKT31 CMA labels
    wget -O OASIS-TRT-20_DKT31_CMA_labels_v2.zip "https://files.osf.io/v1/resources/hs8am/providers/osfstorage/57c1a8ffb83f690201c4a8be/?zip="
    unzip -o OASIS-TRT-20_DKT31_CMA_labels_v2.zip -d OASIS-TRT-20_DKT31_CMA_labels_v2
    rm OASIS-TRT-20_DKT31_CMA_labels_v2.zip

    # Download joint fusion labels
    wget -O OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz "https://osf.io/download/nxg5t/"

%files
    ./src /opt/src
    ./requirements.txt /opt/requirements.txt

%post
    # Install Python dependencies
    cd /opt
    python3 -m pip install -r requirements.txt

    # Install ants_seg_to_nidm
    cd /opt/src/ants_seg_to_nidm
    python3 -m pip install -e .
    python3 -m pip install -r requirements.txt

%environment
    # Add opt to Python path
    export PYTHONPATH=/opt:$PYTHONPATH
    # Add Python packages to path
    export PATH=/usr/local/bin:$PATH

%runscript
    # Execute the Python entry point directly
    python3 /opt/src/run.py "$@"

%help
    ANTs BIDS App 0.1.0

    This container performs ANTs-based segmentation and converts results to NIDM format.

    Version Information:
    - ANTs BIDS App: 0.1.0
    - Python: 3.10
    - Base: Ubuntu 22.04

    Usage:
      singularity run [container] [input_dir] [output_dir] participant [options]

    Example:
      singularity run ants_bidsapp.sif $PWD/inputs/data/BIDS $PWD/outputs/ants participant --participant-label 01 02 03 --session-label 01 --modality T1w