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
import ants
import pkg_resources

# Import local modules
from src.antspy.wrapper import ANTsSegmentation

def get_bids_version():
    """Get the version of the installed bids package."""
    try:
        return pkg_resources.get_distribution('pybids').version
    except pkg_resources.DistributionNotFound:
        # Fallback to a recent stable version if package info not found
        return "1.8.0"

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
        "BIDSVersion": get_bids_version(),
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
    parser.add_argument('--participant-label', '--participant_label', dest='participant_label',
                        help='The label of the participant that should be analyzed. The label corresponds to sub-<participant_label> from the BIDS spec (with or without "sub-" prefix).')
    parser.add_argument('--session-label', '--session_label', dest='session_label',
                        help='The label of the session that should be analyzed. The label corresponds to ses-<session_label> from the BIDS spec (with or without "ses-" prefix).')
    parser.add_argument('--modality', help='Modality to process [default: T1w]',
                        default='T1w')
    parser.add_argument('--prob-threshold', help='Probability threshold for binary mask creation [default: 0.5]',
                        type=float, default=0.5)
    
    # Segmentation method
    parser.add_argument('--method', help='Segmentation method to use [default: fusion]',
                        choices=['quick', 'fusion'], default='fusion')
    
    parser.add_argument('--skip-bids-validation', help='Skip BIDS validation step',
                        action='store_true')
    parser.add_argument('--skip-nidm', help='Skip NIDM conversion step',
                        action='store_true')
    
    # Skip ANTs and process existing results
    parser.add_argument('--skip-ants', help='Skip ANTs segmentation and only run NIDM conversion. Requires --ants-input.',
                        action='store_true')
    parser.add_argument('--ants-input', help='Path to existing ANTs segmentation derivatives. Required if --skip-ants is set.',
                        type=str, default=None)
    parser.add_argument('--nidm-input', help='Path to existing NIDM TTL file to update (will be copied to output before updating).',
                        type=str, default=None)
    
    parser.add_argument('--num-threads', help='Number of threads to use for processing [default: 1]',
                        type=int, default=1)
    parser.add_argument('-v', '--verbose', help='Verbose output',
                        action='store_true')
    parser.add_argument('--version', action='version', 
                        version='ANTs BIDS App v0.1.0')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_ants and not args.ants_input:
        parser.error("--skip-ants requires --ants-input to specify existing ANTs derivatives")
    
    return args

def initialize(args):
    """Initialize the ANTs BIDS app.
    Args:
        args: Command line arguments
    Returns:
        tuple: (layout, segmenter, derivatives_dir, nidm_dir, nidm_input_file, temp_dir)
    """
    # Normalize incoming paths from argparse to Path objects
    args.bids_dir = Path(args.bids_dir)
    args.output_dir = Path(args.output_dir)
    
    # Initialize BIDS Layout
    layout = BIDSLayout(str(args.bids_dir), validate=not args.skip_bids_validation)
    
    # Handle NIDM input file - prioritize command line argument over default location
    nidm_input_file = None
    if args.nidm_input:
        nidm_input_file = Path(args.nidm_input)
        if not nidm_input_file.exists():
            raise FileNotFoundError(f"NIDM input file not found: {nidm_input_file}")
    else:
        # Legacy fallback: check default location in parent NIDM folder
        legacy_nidm_dir = args.bids_dir.parent / "NIDM"
        legacy_nidm_file = legacy_nidm_dir / "nidm.ttl"
        if legacy_nidm_file.exists():
            nidm_input_file = legacy_nidm_file
        
    # Create output directory directly (no ants_bidsapp wrapper)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create the output derivative directory with BIDS-compliant structure
    # Outputs go directly to output_dir/ants-seg/ and output_dir/nidm/
    derivatives_dir = args.output_dir / 'ants-seg'
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset_description.json
    create_dataset_description(derivatives_dir, '0.1.0')
    
    # Create temporary directory
    temp_dir = args.output_dir / 'tmp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize segmentation with appropriate parameters (only if not skipping ANTs)
    segmenter = None
    if not args.skip_ants:
        segmenter = ANTsSegmentation(
            bids_dir=str(args.bids_dir),
            output_dir=str(derivatives_dir),
            temp_dir=str(temp_dir),
            modality=args.modality,
            prob_threshold=args.prob_threshold,
            num_threads=args.num_threads,
            verbose=args.verbose
        )
    
    # Create NIDM output directory
    nidm_dir = args.output_dir / 'nidm'
    nidm_dir.mkdir(parents=True, exist_ok=True)
    
    # If we have an input NIDM file, copy it to the output directory
    if nidm_input_file:
        output_nidm_file = nidm_dir / nidm_input_file.name
        if not output_nidm_file.exists():
            shutil.copy2(nidm_input_file, output_nidm_file)
    
    return layout, segmenter, derivatives_dir, nidm_dir, nidm_input_file, temp_dir

def nidm_conversion(logger, derivatives_dir, nidm_dir, bids_subject, nidm_input_file=None, bids_session=None, verbose=False, input_file=None):
    """Convert ANTs segmentation outputs to NIDM format.
    Args:
        logger: Logger instance
        derivatives_dir (str or Path): Path to ANTs derivatives directory
        nidm_dir (str or Path): Path to NIDM output directory
        bids_subject (str): Subject label (without "sub-" prefix)
        nidm_input_file (Path or None): Optional existing NIDM TTL file to update
        bids_session (str): Session label (without "ses-" prefix)
        verbose (bool): Enable verbose output
        input_file (str or Path): Path to the input T1w file
    Returns:
        bool: True if conversion succeeded, False otherwise
    """
    log_prefix = f"subject {bids_subject}" + (f", session {bids_session}" if bids_session else "")
    
    try:
        # Convert paths to Path objects
        derivatives_dir = Path(derivatives_dir)
        nidm_dir = Path(nidm_dir)
        nidm_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing NIDM file in output directory (may have been copied from input)
        existing_nidm_file = None
        if nidm_input_file:
            # The input file should have been copied to nidm_dir during initialize()
            copied_nidm_file = nidm_dir / nidm_input_file.name
            if copied_nidm_file.exists():
                existing_nidm_file = copied_nidm_file
            elif nidm_input_file.exists():
                # Fallback to original if copy didn't happen
                existing_nidm_file = nidm_input_file
        
        # Define paths to segmentation outputs
        # Files are directly under derivatives_dir (no sub-* folder since BABS runs per-participant)
        if bids_session is None:
            # Single session case - no sub-id subfolder
            seg_path = derivatives_dir / "anat" / f"sub-{bids_subject}_space-orig_dseg.nii.gz"
            stats_dir = derivatives_dir / "stats"
            nidm_file = nidm_dir / f"sub-{bids_subject}_nidm.json-ld"
        else:
            # Multi-session case - no sub-id/ses-id subfolders
            seg_path = derivatives_dir / "anat" / f"sub-{bids_subject}_ses-{bids_session}_space-orig_dseg.nii.gz"
            stats_dir = derivatives_dir / "stats"
            nidm_file = nidm_dir / f"sub-{bids_subject}_ses-{bids_session}_nidm.json-ld"
            
        # Define paths to the statistics files
        label_stats = stats_dir / "antslabelstats.csv"
        brain_vols = stats_dir / "antsbrainvols.csv"
        
        # Check if required files exist
        required_files = [seg_path, label_stats, brain_vols]
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                return False
        
        # If adding to existing NIDM file, output should be combined nidm.ttl
        # Otherwise, output is subject-specific JSON-LD file
        if existing_nidm_file:
            nidm_output = nidm_dir / "nidm.ttl"
        else:
            nidm_output = nidm_file
        
        # Convert all paths to strings for subprocess call
        label_stats_str = str(label_stats.absolute())
        brain_vols_str = str(brain_vols.absolute())
        seg_path_str = str(seg_path.absolute())
        nidm_output_str = str(nidm_output.absolute())
        
        # Construct the command to run ants_seg_to_nidm.py
        cmd = [
            "python", "-m",
            "ants_seg_to_nidm.ants_seg_to_nidm",
            "-f", f"{label_stats_str},{brain_vols_str},{seg_path_str}",
            "-subjid", f"sub-{bids_subject}",
            "-o", nidm_output_str
        ]

        # Add JSON-LD flag if output format is JSON-LD
        if nidm_output_str.endswith('.json-ld'):
            cmd.append("-j")

        logger.info(f"Converting segmentation to NIDM for {log_prefix}")
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Add existing NIDM file if available
        if existing_nidm_file:
            cmd.extend(["--nidm", str(existing_nidm_file.absolute()), "--forcenidm"])
            logger.info(f"Adding data to existing NIDM file: {existing_nidm_file}")
            logger.info(f"Combined NIDM will be written to: {nidm_output_str}")

        # Run the command from the script's directory
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
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
    """Run the participant level analysis for single-session datasets."""
    logger.info("Starting participant level analysis (single-session dataset)")
    
    # Initialize app
    layout, segmenter, derivatives_dir, nidm_dir, nidm_input_file, temp_dir = initialize(args)
    
    # Get subject to process
    available_subjects = layout.get_subjects()
    participant_label = args.participant_label
    # Normalize participant label - strip "sub-" if present for consistency
    if participant_label.startswith('sub-'):
        bids_subject = participant_label[4:]
    else:
        bids_subject = participant_label
    participant_label = f"sub-{bids_subject}"  # Ensure full label for logging
    
    if bids_subject not in available_subjects:
        logger.error(f"Subject {participant_label} not found in dataset")
        return 1
    
    # For participant level: process single session (no session folders expected)
    bids_session = None
    
    logger.info(f"Processing single-session data for subject {participant_label}")
    
    success = True
    
    # Run ANTs segmentation (unless skipped)
    if args.skip_ants:
        logger.info("Skipping ANTs segmentation (--skip-ants flag set)")
        # Use existing ANTs derivatives if provided
        if args.ants_input:
            derivatives_dir = Path(args.ants_input)
            logger.info(f"Using existing ANTs derivatives from: {derivatives_dir}")
    else:
        # Run segmentation
        if segmenter.run_subject(participant_label, None, method=args.method):
            logger.info(f"Segmentation completed for subject {participant_label}")
        else:
            success = False
            logger.error(f"Segmentation failed for subject {participant_label}")
    
    # Convert segmentation to NIDM if requested and segmentation succeeded (or skipped)
    if success and not args.skip_nidm:
        # Get input file path for NIDM conversion (single session)
        input_path = layout.get(subject=bids_subject, suffix=args.modality, extension='nii.gz')
        input_file = input_path[0].path if input_path else None
        
        success = nidm_conversion(
            logger=logger,
            derivatives_dir=derivatives_dir,
            nidm_dir=nidm_dir,
            bids_subject=bids_subject,
            nidm_input_file=nidm_input_file,
            bids_session=bids_session,
            verbose=args.verbose,
            input_file=input_file,
        )
    
    logger.info(f"Participant level analysis complete. Processing {'succeeded' if success else 'failed'}")
    
    # Clean up temporary files
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    return 0 if success else 1

def process_session(args, logger):
    """Run the session level analysis for multi-session datasets.
    
    Note: BABS schedules each session as a separate task, so this processes
    ONE session per task execution.
    """
    logger.info("Starting session level analysis (multi-session dataset)")
    
    # Initialize app
    layout, segmenter, derivatives_dir, nidm_dir, nidm_input_file, temp_dir = initialize(args)
    
    # Get subject to process
    available_subjects = layout.get_subjects()
    participant_label = args.participant_label
    # Normalize participant label - strip "sub-" if present for consistency
    if participant_label.startswith('sub-'):
        bids_subject = participant_label[4:]
    else:
        bids_subject = participant_label
    participant_label = f"sub-{bids_subject}"  # Ensure full label for logging
    
    if bids_subject not in available_subjects:
        logger.error(f"Subject {participant_label} not found in dataset")
        return 1
    
    # For session level: --session-label is required by BABS
    if not args.session_label:
        logger.error("--session-label is required for session level analysis")
        return 1
    
    # Normalize session label
    session_label = args.session_label
    if session_label.startswith('ses-'):
        bids_session = session_label[4:]
    else:
        bids_session = session_label
    session_label = f"ses-{bids_session}"  # Ensure full label for logging
    
    # Validate session exists
    available_sessions = layout.get_sessions(subject=bids_subject)
    if bids_session not in available_sessions:
        logger.error(f"Session {session_label} not found for subject {participant_label}")
        return 1
    
    logger.info(f"Processing session {session_label} for subject {participant_label}")
    
    success = True
    
    # Run ANTs segmentation (unless skipped)
    if args.skip_ants:
        logger.info("Skipping ANTs segmentation (--skip-ants flag set)")
        # Use existing ANTs derivatives if provided
        if args.ants_input:
            derivatives_dir = Path(args.ants_input)
            logger.info(f"Using existing ANTs derivatives from: {derivatives_dir}")
    else:
        # Run segmentation
        if segmenter.run_subject(participant_label, session_label, method=args.method):
            logger.info(f"Segmentation completed for session {session_label}")
        else:
            success = False
            logger.error(f"Segmentation failed for session {session_label}")
    
    # Convert segmentation to NIDM if requested and segmentation succeeded (or skipped)
    if success and not args.skip_nidm:
        # Get input file path for NIDM conversion
        input_path = layout.get(subject=bids_subject, session=bids_session, suffix=args.modality, extension='nii.gz')
        input_file = input_path[0].path if input_path else None
        
        success = nidm_conversion(
            logger=logger,
            derivatives_dir=derivatives_dir,
            nidm_dir=nidm_dir,
            bids_subject=bids_subject,
            nidm_input_file=nidm_input_file,
            bids_session=bids_session,
            verbose=args.verbose,
            input_file=input_file,
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
        else:
            logger.error(f"Unsupported analysis level: {args.analysis_level}")
            return 1
    except Exception as e:
        logger.error(f"Error in {args.analysis_level} level analysis: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
