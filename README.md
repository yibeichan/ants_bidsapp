# ANTs BIDS App

A BIDS App for ANTs-based brain segmentation with NIDM outputs.

## Overview

This BIDS App provides a standardized way to run ANTs-based brain segmentation on BIDS-formatted datasets. It includes preprocessing steps, segmentation, and generates NIDM-compatible outputs for better reproducibility and sharing of results.

## Features

- BIDS-compliant input/output
- ANTs-based brain segmentation
- N4 bias field correction
- Brain extraction
- Tissue segmentation
- NIDM-compatible outputs
- Docker and Singularity support

## Installation

### Container Images

Pre-built images will be available once the app is published to Docker Hub. For now, please build from source (see below).

### From Source

```bash
git clone https://github.com/ReproNim/ants-bidsapp.git
cd ants-bidsapp
pip install -e .
```

### Building Containers from Source

This BIDS App follows standard BIDS Apps practices with a Dockerfile as the primary container definition. For HPC environments without Docker, we also provide a native Singularity definition file.

#### Building with Docker (on systems with Docker installed)

```bash
# Using the setup.py helper script
python setup.py docker

# Or directly with Docker
docker build -t ants-bidsapp:latest .

# Save for transfer to HPC (if needed)
docker save ants-bidsapp:latest -o ants-bidsapp.tar
```

#### Building with Singularity/Apptainer (for HPC environments)

```bash
# Direct build from Singularity definition file
# The --fakeroot flag is required on HPC systems without root access
apptainer build --fakeroot ants-bidsapp.sif Singularity

# Or using the setup.py helper
python setup.py singularity
```

#### Converting Docker to Singularity

If you have a Docker image (either built locally or from a tar file):

```bash
# From a saved Docker tar file
singularity build ants-bidsapp.sif docker-archive://ants-bidsapp.tar

# From local Docker daemon (requires Docker)
singularity build ants-bidsapp.sif docker-daemon://ants-bidsapp:latest
```

## Usage

### Basic Usage

```bash
ants-bidsapp bids_dir output_dir participant --participant-label 01
```

### Advanced Options

```bash
ants-bidsapp bids_dir output_dir participant \
  --participant-label 01 \
  --session-label pre \
  --modality T1w \
  --prob-threshold 0.5 \
  --num-threads 4 \
  --verbose
```

### Command-line Arguments

- `bids_dir`: Path to the BIDS dataset
- `output_dir`: Path where outputs will be stored
- `analysis_level`: Level of the analysis (participant)
- `--participant-label`: Label(s) of the participant(s) to analyze
- `--session-label`: Label(s) of the session(s) to analyze
- `--modality`: Imaging modality to process (default: T1w)
- `--prob-threshold`: Probability threshold for binary mask creation (default: 0.5)
- `--priors`: Paths to prior probability maps for segmentation
- `--skip-nidm`: Skip NIDM conversion step
- `--num-threads`: Number of threads to use for processing (default: 1)
- `--verbose`: Print detailed logs

## Outputs

The app generates the following outputs:

- Segmentation results in BIDS-compatible format
- Probability maps for each tissue class
- Binary masks for each tissue class
- NIDM-compatible outputs for reproducibility

## NIDM Outputs

The app generates NIDM-compatible outputs that can be used with NIDM tools for visualization and sharing of results. The NIDM outputs include:

- Segmentation statistics
- Brain volumes
- Tissue volumes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this BIDS App in your research, please cite:

```
ANTs BIDS App. ReproNim. https://github.com/ReproNim/ants-bidsapp
```