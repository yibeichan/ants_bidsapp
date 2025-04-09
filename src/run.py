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

# Import local modules
from ants.wrapper import ANTsSegmentation
# We'll import from the submodule
sys.path.append(os.path.join(os.path.dirname(__file__), 'ants_seg_to_nidm'))
from ants_seg_to_nidm.ants_seg_to_nidm import ants_seg_to_nidm
import utils

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
                "CodeURL": "https://github.com/yourusername/ants_bidsapp"
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
                        choices=['participant', 'group'])
    
    # Optional arguments
    parser.add_argument('--participant-label', help='The label(s) of the participant(s) that should be analyzed. The label corresponds to sub-<participant_label> from the BIDS spec. If this parameter is not provided, all subjects will be analyzed. Multiple participants can be specified with a space-separated list.', 
                        nargs='+')
    parser.add_argument('--session-label', help='The label(s) of the session(s) that should be analyzed. The label corresponds to ses-<session_label> from the BIDS spec. If this parameter is not provided, all sessions will be analyzed. Multiple sessions can be specified with a space-separated list.',
                        nargs='+')
    parser.add_argument('--modality', help='Modality to process [default: T1w]',
                        default='T1w')
    parser.add_argument('--prob-threshold', help='Probability threshold for binary mask creation [default: 0.5]',
                        type=float, default=0.5)
    parser.add_argument('--priors', help='Paths to prior probability maps for segmentation',
                        nargs='+')
    parser.add_argument('--skip-nidm', help='Skip NIDM conversion step',
                        action='store_true')
    parser.add_argument('--num-threads', help='Number of threads to use for processing [default: 1]',
                        type=int, default=1)
    parser.add_argument('-v', '--verbose', help='Verbose output',
                        action='store_true')
    parser.add_argument('--version', action='version', 
                        version='ANTs BIDS App v0.1.0')
    
    return parser.parse_args()

def validate_bids(bids_dir):
    """Validate BIDS directory structure."""
    # Basic validation - check for key directories and files
    if not os.path.isdir(bids_dir):
        raise ValueError(f"BIDS directory does not exist: {bids_dir}")
    
    # Check for dataset_description.json
    if not os.path.exists(os.path.join(bids_dir, 'dataset_description.json')):
        logging.warning(f"dataset_description.json not found in {bids_dir}. This may not be a valid BIDS dataset.")
    
    # Check for at least one subject directory
    subjects = [d for d in os.listdir(bids_dir) if d.startswith('sub-')]
    if not subjects:
        raise ValueError(f"No subject directories found in {bids_dir}")
    
    # For more comprehensive validation, consider using a BIDS validator tool
    logging.info(f"Found {len(subjects)} subjects in BIDS directory")
    return subjects

def run_participant_level(args, logger):
    """Run the participant level analysis."""
    logger.info("Starting participant level analysis")
    
    # Validate input directory
    subjects = validate_bids(args.bids_dir)
    
    # Filter subjects if participant_label is provided
    if args.participant_label:
        subjects = [s for s in subjects if s.replace('sub-', '') in args.participant_label]
        if not subjects:
            logger.error(f"No matching subjects found for labels: {args.participant_label}")
            return 1
    
    # Create the output derivative directory with BIDS-compliant structure
    derivatives_dir = os.path.join(args.output_dir, 'ants-seg')
    os.makedirs(derivatives_dir, exist_ok=True)
    
    # Create dataset_description.json
    create_dataset_description(derivatives_dir, '0.1.0')
    
    # Create temporary directory
    temp_dir = os.path.join(args.output_dir, 'tmp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize ANTs segmentation
    segmenter = ANTsSegmentation(
        input_path=args.bids_dir,
        output_path=derivatives_dir,
        temp_path=temp_dir,
        priors=args.priors,
        modality=args.modality,
        prob_threshold=args.prob_threshold,
        num_threads=args.num_threads,
        verbose=args.verbose
    )
    
    # Process each subject
    success_count = 0
    for subject in subjects:
        subject_id = subject.replace('sub-', '')
        logger.info(f"Processing subject: {subject_id}")
        
        subject_dir = os.path.join(args.bids_dir, subject)
        sessions = [d for d in os.listdir(subject_dir) if d.startswith('ses-')]
        
        # Filter sessions if session_label is provided
        if args.session_label and sessions:
            sessions = [s for s in sessions if s.replace('ses-', '') in args.session_label]
            if not sessions:
                logger.warning(f"No matching sessions found for subject {subject_id}")
                continue
        
        if sessions:
            # Process each session
            for session in sessions:
                session_id = session.replace('ses-', '')
                if segmenter.run_subject(subject_id, session_id):
                    success_count += 1
                    
                    # Convert segmentation to NIDM if requested
                    if not args.skip_nidm:
                        try:
                            logger.info(f"Converting segmentation to NIDM for subject {subject_id}, session {session_id}")
                            nidm_dir = os.path.join(args.output_dir, 'nidm')
                            os.makedirs(nidm_dir, exist_ok=True)
                            
                            # Define paths to segmentation outputs
                            seg_path = os.path.join(
                                derivatives_dir, 
                                f"sub-{subject_id}", 
                                f"ses-{session_id}", 
                                "anat",
                                f"sub-{subject_id}_ses-{session_id}_space-orig_dseg.nii.gz"
                            )
                            
                            # Define paths to the statistics files
                            stats_dir = os.path.join(derivatives_dir, f"sub-{subject_id}", f"ses-{session_id}", "stats")
                            label_stats = os.path.join(stats_dir, "antslabelstats.csv")
                            brain_vols = os.path.join(stats_dir, "antsbrainvols.csv")
                            
                            # Define output NIDM file path
                            nidm_file = os.path.join(nidm_dir, f"sub-{subject_id}_ses-{session_id}_NIDM.ttl")
                            
                            # Construct the command to run ants_seg_to_nidm.py
                            cmd = [
                                "python", 
                                os.path.join(os.path.dirname(__file__), "ants_seg_to_nidm", "ants_seg_to_nidm.py"),
                                "-f", f"{label_stats},{brain_vols},{seg_path}",
                                "-subjid", subject_id,
                                "-o", nidm_file
                            ]
                            
                            logger.info(f"Running command: {' '.join(cmd)}")
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            
                            if result.returncode != 0:
                                logger.error(f"Error in NIDM conversion: {result.stderr}")
                            else:
                                logger.info(f"NIDM conversion complete: {nidm_file}")
                                logger.debug(f"NIDM conversion output: {result.stdout}")
                                
                        except Exception as e:
                            logger.error(f"Error in NIDM conversion for subject {subject_id}, session {session_id}: {str(e)}")
        else:
            # Process subject without sessions
            if segmenter.run_subject(subject_id):
                success_count += 1
                
                # Convert segmentation to NIDM if requested
                # For the case without sessions
                if not args.skip_nidm:
                    try:
                        logger.info(f"Converting segmentation to NIDM for subject {subject_id}")
                        nidm_dir = os.path.join(args.output_dir, 'nidm')
                        os.makedirs(nidm_dir, exist_ok=True)
                        
                        # Define paths to segmentation outputs
                        seg_path = os.path.join(
                            derivatives_dir, 
                            f"sub-{subject_id}", 
                            "anat",
                            f"sub-{subject_id}_space-orig_dseg.nii.gz"
                        )
                        
                        # Define paths to the statistics files
                        stats_dir = os.path.join(derivatives_dir, f"sub-{subject_id}", "stats")
                        label_stats = os.path.join(stats_dir, "antslabelstats.csv")
                        brain_vols = os.path.join(stats_dir, "antsbrainvols.csv")
                        
                        # Define output NIDM file path
                        nidm_file = os.path.join(nidm_dir, f"sub-{subject_id}_NIDM.ttl")
                        
                        # Construct the command to run ants_seg_to_nidm.py
                        cmd = [
                            "python", 
                            os.path.join(os.path.dirname(__file__), "ants_seg_to_nidm", "ants_seg_to_nidm.py"),
                            "-f", f"{label_stats},{brain_vols},{seg_path}",
                            "-subjid", subject_id,
                            "-o", nidm_file
                        ]
                        
                        logger.info(f"Running command: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            logger.error(f"Error in NIDM conversion: {result.stderr}")
                        else:
                            logger.info(f"NIDM conversion complete: {nidm_file}")
                            logger.debug(f"NIDM conversion output: {result.stdout}")
                            
                    except Exception as e:
                        logger.error(f"Error in NIDM conversion for subject {subject_id}: {str(e)}")
    
    logger.info(f"Participant level analysis complete. Processed {success_count} subjects successfully.")
    
    # Clean up temporary files
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    return 0

def run_group_level(args, logger):
    """Run the group level analysis."""
    logger.info("Starting group level analysis")
    
    # This is a placeholder for group level analysis
    # In a real implementation, you might want to:
    # 1. Collect all segmentation results
    # 2. Create group-level statistics
    # 3. Generate group-level visualizations
    
    logger.info("Group level analysis is not implemented yet.")
    return 0

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
        # Run the appropriate analysis level
        if args.analysis_level == 'participant':
            return run_participant_level(args, logger)
        elif args.analysis_level == 'group':
            return run_group_level(args, logger)
        else:
            logger.error(f"Unknown analysis level: {args.analysis_level}")
            return 1
    except Exception as e:
        logger.error(f"Error in {args.analysis_level} level analysis: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())