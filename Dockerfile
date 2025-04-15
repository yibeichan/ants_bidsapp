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

# Copy the application code
COPY src/ /opt/src/
COPY requirements.txt /opt/requirements.txt
COPY data/ /opt/data/

# Install ants_seg_to_nidm
RUN cd src/ants_seg_to_nidm && \
    python -m pip install -e . && \
    cd /opt && \
    python -m pip install -r requirements.txt && \
    python -m pip install -e .

# Set environment variables
ENV PYTHONPATH=/opt:$PYTHONPATH
ENV PATH=/usr/local/bin:$PATH

# Set the entrypoint
ENTRYPOINT ["python", "/opt/src/run.py"] 