# Example Data

This directory contains example data for testing and demonstration purposes.

## Directory Structure

```
examples/
├── dataset_description.json
├── sub-01/
│   ├── anat/
│   │   └── sub-01_T1w.nii.gz
│   └── sub-01_scans.tsv
└── sub-02/
    ├── anat/
    │   └── sub-02_T1w.nii.gz
    └── sub-02_scans.tsv
```

## Usage

To run the ANTs BIDS App on this example data:

```bash
ants-bidsapp examples output participant --participant-label 01 02
```

## Note

The example data files are placeholders and do not contain actual MRI data. For real usage, replace them with your own BIDS-formatted data. 