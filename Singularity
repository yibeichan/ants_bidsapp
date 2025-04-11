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
        && apt clean && \
        rm -rf /var/lib/apt/lists/*

    # Create symbolic links for python3.10
    ln -s /usr/bin/python3.10 /usr/bin/python3
    ln -s /usr/bin/python3.10 /usr/bin/python

    # Upgrade pip and install basic Python tools
    python3 -m pip install --upgrade pip setuptools wheel

    # Create application directory
    mkdir -p /opt/src

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