# src/ants/wrapper.py
import os
import subprocess
import logging
import ants
import nibabel as nib
import numpy as np
from pathlib import Path

class ANTsSegmentation:
    """
    Wrapper class for ANTs segmentation functionality using ANTsPy.
    This class handles the preparation, execution, and post-processing
    of ANTs-based segmentation for neuroimaging data in BIDS format.
    """
    
    def __init__(self, 
                 bids_dir=None, 
                 output_dir=None,
                 temp_dir=None,
                 priors=None,
                 modality='T1w',
                 prob_threshold=0.5,
                 num_threads=1,
                 verbose=False):
        """Initialize ANTs segmentation.
        
        Parameters
        ----------
        bids_dir : str or Path
            Path to BIDS directory
        output_dir : str or Path
            Path to output directory
        temp_dir : str or Path
            Path to temporary directory
        priors : list
            List of paths to prior probability maps
        modality : str
            Modality to process (default: T1w)
        prob_threshold : float
            Probability threshold for binary mask creation
        num_threads : int
            Number of threads to use
        verbose : bool
            Whether to print detailed logs
        """
        self.bids_dir = Path(bids_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir) if temp_dir else self.output_dir / 'tmp'
        self.priors = priors
        self.modality = modality
        self.prob_threshold = prob_threshold
        self.num_threads = num_threads
        self.verbose = verbose
        
        # Set up logging
        self.logger = logging.getLogger('ants_bidsapp.segmentation')
        
        # Create necessary directories
        if output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
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
        """Perform ANTs segmentation on the given image.
        
        Parameters
        ----------
        image : ants.ANTsImage
            Preprocessed image to segment
        priors : list, optional
            List of prior probability maps (defaults to self.priors)
        prior_weight : float, optional
            Weight for priors (0=initialization only, 0.25-0.5=regularization)
        mrf : float, optional
            MRF smoothing parameter (higher=smoother)
        iterations : int, optional
            Maximum number of iterations
            
        Returns
        -------
        dict
            Dictionary containing segmentation results
        """
        self.logger.info("Starting image segmentation")
        
        # Get mask for segmentation
        mask = ants.get_mask(image)
        
        # Use provided priors or fall back to class-level priors
        use_priors = priors if priors is not None else self.priors
        
        # Use Atropos segmentation
        if use_priors:
            self.logger.debug("Using prior probability maps")
            segmentation = ants.atropos(a=image, 
                                      m=f'[{mrf},1x1x1]',
                                      c=f'[{iterations},0]',
                                      i=use_priors,
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
    
    def _organize_bids_output(self, segmentation, bids_subject, bids_session=None):
        """Organize ANTs outputs in BIDS-compliant format.

        Parameters
        ----------
        segmentation : dict
            Dictionary containing segmentation results
        bids_subject : str
            Subject ID (without "sub-" prefix)
        bids_session : str, optional
            BIDS session label (without "ses-" prefix)
        
        Returns
        -------
        dict
            Dictionary containing tissue and label volumes
        """
        # Set up directories
        session_part = f"_ses-{bids_session}" if bids_session else ""
        bids_subject_dir = self.output_dir / f"sub-{bids_subject}"
        if bids_session:
            bids_subject_dir = bids_subject_dir / f"ses-{bids_session}"

        anat_dir = bids_subject_dir / "anat"
        stats_dir = bids_subject_dir / "stats"
        anat_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)

        # Save probability maps and create binary masks
        tissue_volumes = {}  # Store volumes for brainvols.csv
        label_volumes = {}   # Store volumes for labelstats.csv
        
        # Save probability maps and binary masks
        for idx, prob in enumerate(segmentation['probabilityimages']):
            tissue_type = f"tissue{idx+1}"
            
            # Save probability map
            prob_filename = f"sub-{bids_subject}{session_part}_space-orig_{tissue_type}_probseg.nii.gz"
            prob_path = anat_dir / prob_filename
            ants.image_write(prob, str(prob_path))
            
            # Create and save binary mask
            binary_mask = (prob.numpy() > self.prob_threshold).astype(np.uint8)
            binary_img = ants.from_numpy(binary_mask, origin=prob.origin, 
                                       spacing=prob.spacing, direction=prob.direction)
            
            mask_filename = f"sub-{bids_subject}{session_part}_space-orig_{tissue_type}_mask.nii.gz"
            mask_path = anat_dir / mask_filename
            ants.image_write(binary_img, str(mask_path))
            
            # Calculate tissue volume
            voxel_volume = np.prod(prob.spacing)
            tissue_volumes[tissue_type] = np.sum(binary_mask) * voxel_volume
        
        # Save labeled image
        label_filename = f"sub-{bids_subject}{session_part}_space-orig_dseg.nii.gz"
        label_path = anat_dir / label_filename
        ants.image_write(segmentation['segmentation'], str(label_path))
        
        # Generate antslabelstats.csv
        labelstats_file = stats_dir / "antslabelstats.csv"
        with open(labelstats_file, 'w') as f:
            f.write("Label,Volume\n")  # Header for label stats
            
            # Extract labeled data and calculate volumes
            labels = segmentation['segmentation'].numpy()
            unique_labels = np.unique(labels)
            voxel_volume = np.prod(segmentation['segmentation'].spacing)
            
            for label in unique_labels:
                if label > 0:  # Skip background
                    volume = np.sum(labels == label) * voxel_volume
                    label_volumes[int(label)] = volume
                    f.write(f"{int(label)},{volume:.2f}\n")
        
        # Generate antsbrainvols.csv
        brainvols_file = stats_dir / "antsbrainvols.csv"
        with open(brainvols_file, 'w') as f:
            f.write("Name,Value\n")  # Header for brain volumes
            
            # Calculate total brain volume (all non-zero voxels)
            brain_mask = (labels > 0)
            brain_volume = np.sum(brain_mask) * voxel_volume
            
            # Write brain volumes
            f.write(f"BVOL,{brain_volume:.2f}\n")
            
            # Write tissue volumes (assuming order: CSF, GM, WM)
            if len(tissue_volumes) >= 3:
                f.write(f"CSFVOL,{tissue_volumes['tissue1']:.2f}\n")
                f.write(f"GMVOL,{tissue_volumes['tissue2']:.2f}\n")
                f.write(f"WMVOL,{tissue_volumes['tissue3']:.2f}\n")
        
        self.logger.info("Segmentation results organized in BIDS format")
        return {'tissue_volumes': tissue_volumes, 'label_volumes': label_volumes}

    def save_results(self, segmentation, bids_subject, bids_session=None, output_dir=None):
        """Save segmentation results in BIDS-compatible format and generate files for NIDM conversion.
        
        Parameters
        ----------
        segmentation : dict
            Dictionary containing segmentation results
        bids_subject : str
            Subject identifier
        bids_session : str, optional
            Session identifier
        output_dir : str, optional
            Output directory (defaults to self.output_path)
        """
        if output_dir:
            original_output = self.output_dir
            self.output_dir = output_dir

        try:
            # Organize outputs in BIDS format
            volumes = self._organize_bids_output(segmentation, bids_subject, bids_session)
            self.logger.info("Segmentation results saved successfully")
            return volumes
        finally:
            if output_dir:
                self.output_dir = original_output
        
    def run_subject(self, subject_id, session_label=None):
        """Run ANTs segmentation for a subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (with "sub-" prefix)
        session_label : str, optional
            Session label (with "ses-" prefix)
            
        Returns
        -------
        bool
            True if processing succeeded, False otherwise
        """
        try:
            # Strip prefixes for BIDS queries
            bids_subject = subject_id[4:] if subject_id.startswith('sub-') else subject_id
            if session_label:
                bids_session = session_label[4:] if session_label.startswith('ses-') else session_label
            else:
                bids_session = None
                
            # Construct path to input image
            if bids_session:
                bids_path = self.bids_dir / f"sub-{bids_subject}" / f"ses-{bids_session}" / "anat"
                img_pattern = f"sub-{bids_subject}_ses-{bids_session}_{self.modality}.nii.gz"
            else:
                bids_path = self.bids_dir / f"sub-{bids_subject}" / "anat"
                img_pattern = f"sub-{bids_subject}_{self.modality}.nii.gz"
            
            # Find input image
            input_file = list(bids_path.glob(img_pattern))
            if not input_file:
                self.logger.error(f"No {self.modality} image found for subject {subject_id}")
                return False
            
            input_file = input_file[0]
            self.logger.info(f"Processing {input_file}")
            
            # Load and preprocess image
            img = self.load_image(str(input_file))
            preprocessed_img = self.preprocess_image(img)
            
            # Run segmentation
            segmentation = self.segment_image(preprocessed_img)
            if not segmentation:
                return False
            
            # Save results and get volumes
            volumes = self.save_results(segmentation, bids_subject, bids_session)
            
            # Log volume information
            if volumes:
                self.logger.info("Segmentation volumes:")
                if 'tissue_volumes' in volumes:
                    for tissue, volume in volumes['tissue_volumes'].items():
                        self.logger.info(f"  {tissue}: {volume:.2f} mm³")
                if 'label_volumes' in volumes:
                    for label, volume in volumes['label_volumes'].items():
                        self.logger.info(f"  Label {label}: {volume:.2f} mm³")
            
            self.logger.info(f"Segmentation complete for subject {subject_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing subject {subject_id}: {str(e)}")
            return False
        