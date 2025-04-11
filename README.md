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

### Using Docker

```bash
docker pull repro/ants-bidsapp
```

### Using Singularity/Apptainer

```bash
singularity pull ants-bidsapp.sif docker://repro/ants-bidsapp
```

### From Source

```bash
git clone https://github.com/ReproNim/ants-bidsapp.git
cd ants-bidsapp
pip install -e .
```

### Building Containers from Source

You can build Docker and Singularity containers directly from the source code using the setup.py script:

```bash
# Build Docker container
python setup.py docker

# Build Singularity container
python setup.py singularity

# Build both Docker and Singularity containers
python setup.py containers
```

For Singularity builds on cluster environments, you may need to use the `--fakeroot` option:

```bash
# On a cluster with Apptainer
apptainer build --fakeroot ants-bidsapp.sif Singularity

# Or with Singularity
singularity build --fakeroot ants-bidsapp.sif Singularity
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