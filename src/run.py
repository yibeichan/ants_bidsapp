#!/usr/bin/env python3
# src/run.py

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import shutil
from datetime import datetime
import subprocess
from bids import BIDSLayout

# Import local modules
from src.ants.wrapper import ANTsSegmentation

def setup_logger(log_dir, verbose=False):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"ants_bidsapp-{timestamp}.log")
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('ants_bidsapp')

def create_dataset_description(output_dir, app_version):
    """Create a dataset_description.json file in the output directory."""
    dataset_description = {
        "Name": "ANTs segmentation derivatives",
        "BIDSVersion": "1.7.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "ANTs BIDS App",
                "Version": app_version,
                "CodeURL": "https://github.com/ReproNim/ants_bidsapp"
            }
        ],
        "HowToAcknowledge": "Please cite the ANTs segmentation tool and the NIDM standard in your publications."
    }
    
    with open(os.path.join(output_dir, 'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, indent=4)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ANTs Segmentation BIDS App with NIDM outputs')
    
    # Required arguments
    parser.add_argument('bids_dir', help='The directory with the input dataset formatted according to the BIDS standard.')
    parser.add_argument('output_dir', help='The directory where the output files should be stored.')
    parser.add_argument('analysis_level', help='Level of the analysis that will be performed.',
                        choices=['participant', 'session'])
    
    # Optional arguments
    parser.add_argument('--participant-label', help='The label of the participant that should be analyzed. The label corresponds to sub-<participant_label> from the BIDS spec.')
    parser.add_argument('--session-label', help='The label(s) of the session(s) that should be analyzed. The label corresponds to ses-<session_label> from the BIDS spec. If this parameter is not provided, all sessions will be analyzed. Multiple sessions can be specified with a space-separated list.',
                        nargs='+')
    parser.add_argument('--modality', help='Modality to process [default: T1w]',
                        default='T1w')
    parser.add_argument('--prob-threshold', help='Probability threshold for binary mask creation [default: 0.5]',
                        type=float, default=0.5)
    parser.add_argument('--priors', help='Paths to prior probability maps for segmentation',
                        nargs='+')
    parser.add_argument('--skip-bids-validation', help='Skip BIDS validation step',
                        action='store_true')
    parser.add_argument('--skip-nidm', help='Skip NIDM conversion step',
                        action='store_true')
    
    parser.add_argument('--num-threads', help='Number of threads to use for processing [default: 1]',
                        type=int, default=1)
    parser.add_argument('-v', '--verbose', help='Verbose output',
                        action='store_true')
    parser.add_argument('--version', action='version', 
                        version='ANTs BIDS App v0.1.0')
    
    return parser.parse_args()

def initialize(args):
    """Initialize the ANTs BIDS app.
    Args:
        args: Command line arguments
    Returns:
        tuple: (layout, segmenter, derivatives_dir, nidm_dir, temp_dir)
    """
    # Initialize BIDS Layout
    layout = BIDSLayout(args.bids_dir, validate=not args.skip_bids_validation)
    
    # Create output directories
    output_dir = Path(args.output_dir) / 'ants_bidsapp'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the output derivative directory with BIDS-compliant structure
    derivatives_dir = output_dir / 'ants-seg'
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset_description.json
    create_dataset_description(derivatives_dir, '0.1.0')
    
    # Create temporary directory
    temp_dir = output_dir / 'tmp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ANTs segmentation
    segmenter = ANTsSegmentation(
        bids_dir=args.bids_dir,
        output_dir=derivatives_dir,
        temp_dir=temp_dir,
        priors=args.priors,
        modality=args.modality,
        prob_threshold=args.prob_threshold,
        num_threads=args.num_threads,
        verbose=args.verbose
    )

    # Create NIDM output directory
    nidm_dir = output_dir / 'nidm'
    
    return layout, segmenter, derivatives_dir, nidm_dir, temp_dir

def nidm_conversion(logger, derivatives_dir, nidm_dir, bids_subject, bids_session=None, verbose=False):
    """Convert ANTs segmentation outputs to NIDM format.
    Args:
        logger: Logger instance
        derivatives_dir (str or Path): Path to ANTs derivatives directory
        nidm_dir (str or Path): Path to NIDM output directory
        bids_subject (str): Subject label (without "sub-" prefix)
        bids_session (str): Session label (without "ses-" prefix)
        verbose (bool): Enable verbose output
    Returns:
        bool: True if conversion succeeded, False otherwise
    """
    try:
        # Convert paths to Path objects
        derivatives_dir = Path(derivatives_dir)
        nidm_dir = Path(nidm_dir)
        nidm_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths to segmentation outputs
        if bids_session is None:
            # Single session case
            subject_dir = derivatives_dir / f"sub-{bids_subject}"
            seg_path = subject_dir / "anat" / f"sub-{bids_subject}_space-orig_dseg.nii.gz"
            stats_dir = subject_dir / "stats"
            log_prefix = f"subject {bids_subject}"
        else:
            # Multi-session case
            subject_dir = derivatives_dir / f"sub-{bids_subject}" / f"ses-{bids_session}"
            seg_path = subject_dir / "anat" / f"sub-{bids_subject}_ses-{bids_session}_space-orig_dseg.nii.gz"
            stats_dir = subject_dir / "stats"
            log_prefix = f"subject {bids_subject}, session {bids_session}"
        
        # Define paths to the statistics files
        label_stats = stats_dir / "antslabelstats.csv"
        brain_vols = stats_dir / "antsbrainvols.csv"
        
        # Check if required files exist
        required_files = [seg_path, label_stats, brain_vols]
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                return False
        
        # Construct the command to run ants_seg_to_nidm.py
        cmd = [
            "python", 
            str(Path(__file__).parent / "ants_seg_to_nidm" / "ants_seg_to_nidm" / "ants_seg_to_nidm.py"),
            "-f", f"{label_stats},{brain_vols},{seg_path}",
            "-subjid", f"sub-{bids_subject}",
            "-o", str(nidm_dir),
            "-j"  # Output in JSON-LD format
        ]
        
        logger.info(f"Converting segmentation to NIDM for {log_prefix}")
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error in NIDM conversion: {result.stderr}")
            return False
        else:
            logger.info(f"NIDM conversion complete for {log_prefix}")
            if verbose:
                logger.debug(f"NIDM conversion output: {result.stdout}")
            return True
            
    except Exception as e:
        logger.error(f"Error in NIDM conversion for {log_prefix}: {str(e)}")
        return False

def process_participant(args, logger):
    """Run the participant level analysis."""
    logger.info("Starting participant level analysis")
    
    # Initialize app
    layout, segmenter, derivatives_dir, nidm_dir, temp_dir = initialize(args)
    
    # Get subject to process
    available_subjects = layout.get_subjects()
    participant_label = args.participant_label
    if not participant_label.startswith('sub-'):
        participant_label = f"sub-{participant_label}"
    
    bids_subject = participant_label[4:]  # Strip "sub-" for BIDS query
    if bids_subject not in available_subjects:
        logger.error(f"Subject {participant_label} not found in dataset")
        return 1
    
    success = False  # Initialize success flag
    # Process subject without sessions
    if segmenter.run_subject(participant_label):
        success = True
        
        # Convert segmentation to NIDM if requested
        if not args.skip_nidm:
            success = success and nidm_conversion(
                logger,
                derivatives_dir,
                nidm_dir,
                bids_subject,  # Pass without "sub-" prefix
                verbose=args.verbose
            )
    
    logger.info(f"Participant level analysis complete. Processing {'succeeded' if success else 'failed'}")
    
    # Clean up temporary files
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    return 0 if success else 1

def process_session(args, logger):
    """Run the session level analysis."""
    logger.info("Starting session level analysis")
    
    # Initialize app
    layout, segmenter, derivatives_dir, nidm_dir, temp_dir = initialize(args)
    
    # Get subject to process
    available_subjects = layout.get_subjects()
    participant_label = args.participant_label
    if not participant_label.startswith('sub-'):
        participant_label = f"sub-{participant_label}"
    
    bids_subject = participant_label[4:]  # Strip "sub-" for BIDS query
    if bids_subject not in available_subjects:
        logger.error(f"Subject {participant_label} not found in dataset")
        return 1

    # Validate that the session exists
    available_sessions = layout.get_sessions(subject=bids_subject)
    session_label = args.session_label
    if not session_label.startswith('ses-'):
        session_label = f"ses-{session_label}"
    
    bids_session = session_label[4:]  # Strip "ses-" for BIDS query
    if bids_session not in available_sessions:
        logger.error(f"Session {session_label} not found for subject {participant_label}")
        return 1
    
    success = False  # Initialize success flag
    if segmenter.run_subject(participant_label, session_label):
        success = True
        if not args.skip_nidm:
            success = success and nidm_conversion(
                logger,
                derivatives_dir,
                nidm_dir,
                bids_subject,  # Pass without "sub-" prefix
                bids_session,  # Pass without "ses-" prefix
                args.verbose
            )
    
    logger.info(f"Session level analysis complete. Processing {'succeeded' if success else 'failed'}")
    
    # Clean up temporary files
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    return 0 if success else 1

def main():
    """Main function to coordinate the workflow."""
    args = parse_arguments()
    
    # Set up logging
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = setup_logger(log_dir, args.verbose)
    
    logger.info("Starting ANTs BIDS App")
    logger.info(f"BIDS directory: {args.bids_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Analysis level: {args.analysis_level}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.analysis_level == 'participant':
            return process_participant(args, logger)
        elif args.analysis_level == 'session':
            return process_session(args, logger)
    except Exception as e:
        logger.error(f"Error in {args.analysis_level} level analysis: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())