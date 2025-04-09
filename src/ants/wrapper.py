# src/ants/wrapper.py
import os
import subprocess
import logging
import ants
import nibabel as nib
import numpy as np

class ANTsSegmentation:
    """
    Wrapper class for ANTs segmentation functionality using ANTsPy.
    This class handles the preparation, execution, and post-processing
    of ANTs-based segmentation for neuroimaging data in BIDS format.
    """
    
    def __init__(self, 
                 input_path=None, 
                 output_path=None, 
                 temp_path=None, 
                 priors=None,
                 modality='T1w',
                 prob_threshold=0.5,
                 num_threads=1,
                 verbose=False):
        """
        Initialize the ANTs segmentation wrapper.
        
        Parameters:
        -----------
        input_path : str
            Path to the input BIDS dataset
        output_path : str
            Path where segmentation outputs will be saved
        temp_path : str
            Path for temporary files
        priors : list
            List of paths to prior probability maps for segmentation
        modality : str
            Imaging modality to process (default: 'T1w')
        prob_threshold : float
            Probability threshold for binary mask creation
        num_threads : int
            Number of threads to use for processing
        verbose : bool
            Whether to print detailed logs
        """
        self.input_path = input_path
        self.output_path = output_path
        self.temp_path = temp_path or os.path.join(output_path, 'tmp')
        self.priors = priors
        self.modality = modality
        self.prob_threshold = prob_threshold
        self.num_threads = num_threads
        
        # Configure logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ANTsSegmentation')
        
        # Create necessary directories
        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
    
    def load_image(self, image_path):
        """
        Load an image using ANTsPy.
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
            
        Returns:
        --------
        ants.ANTsImage
            Loaded ANTs image
        """
        self.logger.info(f"Loading image: {image_path}")
        try:
            img = ants.image_read(image_path)
            return img
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {str(e)}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess an ANTs image for segmentation.
        
        Parameters:
        -----------
        image : ants.ANTsImage
            Input image to preprocess
            
        Returns:
        --------
        ants.ANTsImage
            Preprocessed image
        """
        self.logger.info("Preprocessing image")
        
        # N4 bias field correction
        self.logger.debug("Applying N4 bias field correction")
        corrected = ants.n4_bias_field_correction(image)
        
        # Brain extraction
        self.logger.debug("Performing brain extraction")
        brain_mask = ants.get_mask(corrected)
        brain_image = ants.mask_image(corrected, brain_mask)
        
        return brain_image
    
    def segment_image(self, image, priors=None, prior_weight=0.25, mrf=0.2, iterations=10):
        """
        Perform ANTs segmentation on the given image.
        
        Parameters:
        -----------
        image : ants.ANTsImage
            Preprocessed image to segment
        priors : list, optional
            List of prior probability maps
        prior_weight : float, optional
            Weight for priors (0=initialization only, 0.25-0.5=regularization)
        mrf : float, optional
            MRF smoothing parameter (higher=smoother)
        iterations : int, optional
            Maximum number of iterations
            
        Returns:
        --------
        dict
            Dictionary containing segmentation results
        """
        self.logger.info("Starting image segmentation")
        
        # Get mask for segmentation
        mask = ants.get_mask(image)
        
        # Use Atropos segmentation
        if priors:
            self.logger.debug("Using provided prior probability maps")
            segmentation = ants.atropos(a=image, 
                                        m=f'[{mrf},1x1x1]',
                                        c=f'[{iterations},0]',
                                        i=priors,
                                        x=mask,
                                        priorweight=prior_weight)
        else:
            self.logger.debug("Using kmeans initialization for segmentation")
            segmentation = ants.atropos(a=image, 
                                        m=f'[{mrf},1x1x1]',
                                        c=f'[{iterations},0]',
                                        i='Kmeans[3]',
                                        x=mask)
        
        return segmentation
    
    def save_results(self, segmentation, subject_id, session_id=None, output_dir=None):
        """
        Save segmentation results in BIDS-compatible format.
        
        Parameters:
        -----------
        segmentation : dict
            Dictionary containing segmentation results
        subject_id : str
            Subject identifier
        session_id : str, optional
            Session identifier
        output_dir : str, optional
            Output directory (defaults to self.output_path)
        """
        output_dir = output_dir or self.output_path
        
        # Create subject-specific output directory
        if session_id:
            subj_dir = os.path.join(output_dir, f"sub-{subject_id}", f"ses-{session_id}")
        else:
            subj_dir = os.path.join(output_dir, f"sub-{subject_id}")
        
        anat_dir = os.path.join(subj_dir, "anat")
        if not os.path.exists(anat_dir):
            os.makedirs(anat_dir)
        
        # Create stats directory
        stats_dir = os.path.join(subj_dir, "stats")
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
        
        # Save labeled image
        self.logger.info(f"Saving segmentation results to {anat_dir}")
        
        # Save probability maps
        for idx, prob in enumerate(segmentation['probabilityimages']):
            tissue_type = f"tissue{idx+1}"
            
            # BIDS-compatible filename
            if session_id:
                prob_filename = f"sub-{subject_id}_ses-{session_id}_space-orig_{tissue_type}_probseg.nii.gz"
            else:
                prob_filename = f"sub-{subject_id}_space-orig_{tissue_type}_probseg.nii.gz"
            
            prob_path = os.path.join(anat_dir, prob_filename)
            ants.image_write(prob, prob_path)
            
            # Create binary mask using threshold
            binary_mask = (prob.numpy() > self.prob_threshold).astype(np.uint8)
            binary_img = ants.from_numpy(binary_mask, origin=prob.origin, 
                                        spacing=prob.spacing, direction=prob.direction)
            
            if session_id:
                mask_filename = f"sub-{subject_id}_ses-{session_id}_space-orig_{tissue_type}_mask.nii.gz"
            else:
                mask_filename = f"sub-{subject_id}_space-orig_{tissue_type}_mask.nii.gz"
            
            mask_path = os.path.join(anat_dir, mask_filename)
            ants.image_write(binary_img, mask_path)
        
        # Save labeled image
        if session_id:
            label_filename = f"sub-{subject_id}_ses-{session_id}_space-orig_dseg.nii.gz"
        else:
            label_filename = f"sub-{subject_id}_space-orig_dseg.nii.gz"
        
        label_path = os.path.join(anat_dir, label_filename)
        ants.image_write(segmentation['segmentation'], label_path)
        
        # Generate statistics files for NIDM conversion
        # 1. antslabelstats.csv - Label statistics
        labelstats_file = os.path.join(stats_dir, "antslabelstats.csv")
        with open(labelstats_file, 'w') as f:
            f.write("Label,Volume\n")  # Header for label stats
            
            # Extract labeled data and calculate volumes
            labels = segmentation['segmentation'].numpy()
            unique_labels = np.unique(labels)
            voxel_volume = np.prod(segmentation['segmentation'].spacing)
            
            for label in unique_labels:
                if label > 0:  # Skip background
                    volume = np.sum(labels == label) * voxel_volume
                    f.write(f"{int(label)},{volume:.2f}\n")
        
        # 2. antsbrainvols.csv - Brain volumes
        brainvols_file = os.path.join(stats_dir, "antsbrainvols.csv")
        with open(brainvols_file, 'w') as f:
            f.write("Name,Value\n")  # Header for brain volumes
            
            # Calculate total brain volume (all non-zero voxels)
            brain_mask = (labels > 0)
            brain_volume = np.sum(brain_mask) * voxel_volume
            
            # Calculate tissue volumes based on probability maps
            # Assuming first probability map is CSF, second is GM, third is WM
            # This might need adjustment based on your specific segmentation
            if len(segmentation['probabilityimages']) >= 3:
                csf_volume = np.sum(segmentation['probabilityimages'][0].numpy() > self.prob_threshold) * voxel_volume
                gm_volume = np.sum(segmentation['probabilityimages'][1].numpy() > self.prob_threshold) * voxel_volume
                wm_volume = np.sum(segmentation['probabilityimages'][2].numpy() > self.prob_threshold) * voxel_volume
                
                f.write(f"BVOL,{brain_volume:.2f}\n")
                f.write(f"CSFVOL,{csf_volume:.2f}\n")
                f.write(f"GMVOL,{gm_volume:.2f}\n")
                f.write(f"WMVOL,{wm_volume:.2f}\n")
            else:
                # If we don't have enough probability maps, just write brain volume
                f.write(f"BVOL,{brain_volume:.2f}\n")
        
        self.logger.info("Segmentation results saved successfully")
        
    def run_subject(self, subject_id, session_id=None):
        """
        Run the full segmentation pipeline for a subject.
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        session_id : str, optional
            Session identifier
            
        Returns:
        --------
        bool
            True if processing completed successfully
        """
        self.logger.info(f"Processing subject: {subject_id}" + 
                        (f", session: {session_id}" if session_id else ""))
        
        try:
            # Construct path to the input image
            if session_id:
                bids_path = os.path.join(self.input_path, f"sub-{subject_id}", 
                                         f"ses-{session_id}", "anat")
                img_pattern = f"sub-{subject_id}_ses-{session_id}_{self.modality}.nii.gz"
            else:
                bids_path = os.path.join(self.input_path, f"sub-{subject_id}", "anat")
                img_pattern = f"sub-{subject_id}_{self.modality}.nii.gz"
            
            img_path = os.path.join(bids_path, img_pattern)
            
            if not os.path.exists(img_path):
                self.logger.error(f"Input image not found: {img_path}")
                return False
            
            # Load the image
            img = self.load_image(img_path)
            
            # Preprocess the image
            preprocessed_img = self.preprocess_image(img)
            
            # Segment the image
            segmentation = self.segment_image(preprocessed_img, self.priors)
            
            # Save the results
            self.save_results(segmentation, subject_id, session_id)
            
            self.logger.info(f"Subject {subject_id} processed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing subject {subject_id}: {str(e)}")
            return False
    
    def run_dataset(self):
        """
        Run the segmentation pipeline on all subjects in the BIDS dataset.
        
        Returns:
        --------
        dict
            Dictionary with processing status for each subject
        """
        self.logger.info(f"Processing BIDS dataset at {self.input_path}")
        
        # Get list of subjects
        subjects = [d for d in os.listdir(self.input_path) if d.startswith('sub-')]
        
        results = {}
        
        for subject in subjects:
            subject_id = subject.replace('sub-', '')
            subject_dir = os.path.join(self.input_path, subject)
            
            # Check for sessions
            sessions = [d for d in os.listdir(subject_dir) if d.startswith('ses-')]
            
            if sessions:
                # Process each session
                for session in sessions:
                    session_id = session.replace('ses-', '')
                    success = self.run_subject(subject_id, session_id)
                    results[f"{subject_id}/{session_id}"] = success
            else:
                # Process subject without session
                success = self.run_subject(subject_id)
                results[subject_id] = success
        
        # Log summary
        success_count = sum(1 for v in results.values() if v)
        self.logger.info(f"Processing complete. Successful: {success_count}/{len(results)}")
        
        return results