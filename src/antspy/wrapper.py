# src/ants/wrapper.py
import os
import subprocess
import logging
import ants
import nibabel as nib
import numpy as np
from pathlib import Path
import glob
from natsort import natsorted

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
        modality : str
            Modality to process (default: T1w)
        prob_threshold : float
            Probability threshold for binary mask creation
        num_threads : int
            Number of threads to use
        verbose : bool
            Whether to print detailed logs
        """
        self.bids_dir = Path(bids_dir) if bids_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.temp_dir = Path(temp_dir) if temp_dir else (self.output_dir / 'tmp' if output_dir else None)
        
        # Set container paths
        self.template_dir = Path('/opt/data/OASIS-30_Atropos_template')
        self.atlas_dir = Path('/opt/data/OASIS-TRT-20_brains')
        self.labels_dir = Path('/opt/data/OASIS-TRT-20_DKT31_CMA_labels_v2')
        self.template_labels_path = Path('/opt/data/OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz')
        
        self.modality = modality
        self.prob_threshold = prob_threshold
        self.num_threads = num_threads
        self.verbose = verbose
        
        # Set up logging
        self.logger = logging.getLogger('ants_bidsapp.segmentation')
        
        # Validate required directories
        if not self.bids_dir or not self.bids_dir.exists():
            raise ValueError(f"BIDS directory not found: {self.bids_dir}")
        if not self.output_dir:
            raise ValueError("Output directory must be specified")
        if not self.template_dir.exists():
            raise ValueError(f"Template directory not found: {self.template_dir}")
        
        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
        # Load templates
        self.templates = self._load_templates()
        
        # Validate atlas directories
        if not self.atlas_dir.exists() or not self.labels_dir.exists():
            self.logger.warning("Atlas directories not found. Only 'quick' segmentation method will be available.")
            
    def _load_templates(self):
        """Load OASIS-30 Atropos templates.
        
        Returns
        -------
        dict
            Dictionary containing loaded template images
        """
        if not self.template_dir or not self.template_dir.exists():
            raise ValueError(f"Template directory not found: {self.template_dir}")
            
        templates = {}
        try:
            # Load main templates
            templates['T_template'] = ants.image_read(str(self.template_dir / 'T_template0.nii.gz'))
            templates['BrainCerebellum'] = ants.image_read(str(self.template_dir / 'T_template0_BrainCerebellum.nii.gz'))
            templates['ProbabilityMask'] = ants.image_read(str(self.template_dir / 'T_template0_BrainCerebellumProbabilityMask.nii.gz'))
            templates['ExtractionMask'] = ants.image_read(str(self.template_dir / 'T_template0_BrainCerebellumExtractionMask.nii.gz'))
            
            # Load tissue priors
            templates['priors'] = []
            priors_dir = self.template_dir / 'Priors2'
            for i in range(1, 7):
                prior = ants.image_read(str(priors_dir / f'priors{i}.nii.gz'))
                templates['priors'].append(prior)
                
            self.logger.info("Successfully loaded all templates")
            return templates
            
        except Exception as e:
            self.logger.error(f"Failed to load templates: {str(e)}")
            raise
    
    def load_image(self, image_path):
        """Load an image using ANTs.
        
        Parameters
        ----------
        image_path : str or Path
            Path to the image file
            
        Returns
        -------
        ants.ANTsImage or None
            Loaded image or None if loading fails
        """
        try:
            self.logger.info(f"Loading image: {image_path}")
            img = ants.image_read(str(image_path))
            if img is None:
                self.logger.error(f"Failed to load image {image_path}")
                return None
            return img
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None
    
    def preprocess_image(self, image):
        """
        Preprocess an ANTs image for segmentation.
        
        Parameters:
        -----------
        image : ants.ANTsImage
            Input image to preprocess
            
        Returns
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
    
    def segment_with_jointfusion(self, target_image, atlas_images, atlas_labels=None, search_radius=3, patch_radius=2, beta=4.0, rho=0.01, use_correlation=False, non_negative=False, no_zeroes=False, max_lab_plus_one=False):
        """Perform ANTs JointLabelFusion segmentation.
        
        Parameters
        ----------
        target_image : ants.ANTsImage
            Target image to segment
        atlas_images : list of ants.ANTsImage
            List of atlas images registered to target space
        atlas_labels : list of ants.ANTsImage, optional  
            List of atlas label images registered to target space.
            If not provided, will perform intensity fusion only.
        search_radius : int, optional
            Search radius for patch matching (default=3)
        patch_radius : int, optional
            Patch radius for local intensity similarity (default=2)
        beta : float, optional
            Weight sharpness parameter for patch similarity (default=4.0)
        rho : float, optional
            Gradient step size (default=0.01)
        use_correlation : bool, optional
            Use correlation metric (default=False)
        non_negative : bool, optional
            Use non-negative weights (default=False)
        no_zeroes : bool, optional
            Exclude zeroes from computations (default=False)
        max_lab_plus_one : bool, optional
            Add max label plus one to output (default=False)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'segmentation': ANTsImage of segmentation (None if no atlas_labels provided)
            - 'intensity': ANTsImage of intensity fusion result
            - 'probabilityimages': list of probability maps (empty if no atlas_labels provided)
            - 'segmentation_numbers': list of label numbers (empty if no atlas_labels provided)
        """
        self.logger.info("Starting JointLabelFusion segmentation")
        
        # Get mask for segmentation
        mask = ants.get_mask(target_image)
        
        # Run JointLabelFusion
        if atlas_labels is None:
            self.logger.info("No atlas labels provided, performing intensity fusion only")
            intensity = ants.joint_label_fusion(
                target_image=target_image,
                target_image_mask=mask,
                atlas_list=atlas_images,
                label_list=None,
                beta=beta,
                rad=patch_radius,
                r_search=search_radius,
                rho=rho,
                usecor=use_correlation,
                nonnegative=non_negative,
                no_zeroes=no_zeroes,
                max_lab_plus_one=max_lab_plus_one,
                verbose=self.verbose
            )[1]  # Only take intensity result
            return {
                'segmentation': None,
                'intensity': intensity,
                'probabilityimages': [],
                'segmentation_numbers': []
            }
        else:
            seg, intensity, probimgs, labels = ants.joint_label_fusion(
                target_image=target_image,
                target_image_mask=mask,
                atlas_list=atlas_images,
                label_list=atlas_labels,
                beta=beta,
                rad=patch_radius,
                r_search=search_radius,
                rho=rho,
                usecor=use_correlation,
                nonnegative=non_negative,
                no_zeroes=no_zeroes,
                max_lab_plus_one=max_lab_plus_one,
                verbose=self.verbose
            )
            
            return {
                'segmentation': seg,
                'intensity': intensity,
                'probabilityimages': probimgs,
                'segmentation_numbers': labels
            }

    def segment_image(self, image, method='fusion', **kwargs):
        """Perform ANTs segmentation on the given image.
        
        Parameters
        ----------
        image : ants.ANTsImage
            Input image to segment
        method : str, optional
            Segmentation method to use:
            - 'quick': Use pre-computed labels in template space
            - 'fusion': Use JointLabelFusion with atlas images (default)
        **kwargs : dict
            Additional arguments passed to specific segmentation method
            
        Returns
        -------
        dict
            Dictionary containing segmentation results
        """
        # Run cortical thickness to get N4 correction and brain mask
        self.logger.info("Running cortical thickness pipeline for preprocessing")
        thickness_results = self.compute_cortical_thickness(image, self.template_dir)
        
        # Create masked brain image (equivalent to MultiplyImages node)
        self.logger.info("Creating masked brain image")
        masked_brain = thickness_results['BrainSegmentationN4'] * thickness_results['BrainExtractionMask']
        
        if method == 'quick':
            # Use pre-computed labels in template space
            if not self.template_labels_path.exists():
                raise ValueError(f"Template labels not found: {self.template_labels_path}")
                
            template_labels = ants.image_read(str(self.template_labels_path))
            
            # Apply template-to-subject transforms to labels (equivalent to transformer_nn node)
            warped_labels = ants.apply_transforms(
                fixed=thickness_results['BrainSegmentationN4'],  # Use N4 image as reference
                moving=template_labels,
                transformlist=[
                    thickness_results['TemplateToSubject1GenericAffine'],  # Affine transform
                    thickness_results['TemplateToSubject0Warp']  # Warp transform
                ],
                interpolator='NearestNeighbor',
                dimension=3,
                invert_transform_flags=[False, False]
            )
            
            return {
                'segmentation': warped_labels,
                'BrainSegmentationN4': thickness_results['BrainSegmentationN4'],
                'BrainExtractionMask': thickness_results['BrainExtractionMask']
            }
            
        elif method == 'fusion':
            if not self.atlas_dir.exists() or not self.labels_dir.exists():
                raise ValueError("Atlas directories not found. Cannot use 'fusion' method.")
                
            # Get all atlas T1s and labels
            T1s = natsorted(glob.glob(str(self.atlas_dir / '*.nii.gz')))
            labels = natsorted(glob.glob(str(self.labels_dir / '*.nii.gz')))
            
            if not T1s or not labels:
                raise ValueError("No atlas images or labels found in package data directories")
                
            # Load atlas images and labels
            atlas_images = [ants.image_read(str(path)) for path in T1s]
            atlas_labels = [ants.image_read(str(path)) for path in labels]
            
            # Register each atlas to the masked brain (equivalent to Registration node)
            warped_images = []
            warped_labels = []
            
            for atlas_img, atlas_lab in zip(atlas_images, atlas_labels):
                # Register atlas to target using exact nipype parameters
                reg = ants.registration(
                    fixed=masked_brain,
                    moving=atlas_img,
                    dimension=3,
                    convergence_threshold=[1e-06, 1e-06, 1e-06],
                    convergence_window_size=[20, 20, 10],
                    metric=["Mattes", "Mattes", "CC"],
                    metric_weight=[1, 1, 1],
                    radius_or_number_of_bins=[56, 56, 4],
                    transforms=["Rigid", "Affine", "SyN"],
                    transform_parameters=[(0.05,), (0.08,), (0.1, 3.0, 0.0)],
                    number_of_iterations=[[100, 100], [100, 100], [100, 70, 50, 20]],
                    sampling_strategy=["Regular", "Regular", "None"],
                    sampling_percentage=[0.25, 0.25, 1],
                    smoothing_sigmas=[[2, 1], [1, 0], [3, 2, 1, 0]],
                    sigma_units=["vox", "vox", "vox"],
                    shrink_factors=[[2, 1], [2, 1], [8, 4, 2, 1]],
                    winsorize_upper_quantile=0.995,
                    winsorize_lower_quantile=0.005,
                    use_estimate_learning_rate_once=[True, True, True],
                    use_histogram_matching=[True, True, True],
                    collapse_output_transforms=True,
                    write_composite_transform=True,
                    output_transform_prefix="output_",
                    output_warped_image=True,
                    interpolation="LanczosWindowedSinc",
                    float=True,
                    initial_moving_transform_com=0,
                    num_threads=self.num_threads if self.num_threads > 1 else None
                )
                
                # Apply transforms to atlas labels (equivalent to transformer_nn node)
                warped_lab = ants.apply_transforms(
                    fixed=masked_brain,
                    moving=atlas_lab,
                    transformlist=reg['fwdtransforms'],
                    interpolator='NearestNeighbor',
                    dimension=3
                )
                
                warped_images.append(reg['warpedmovout'])
                warped_labels.append(warped_lab)
            
            # Perform joint label fusion (equivalent to AntsJointFusion node)
            fusion = ants.joint_label_fusion(
                target_image=masked_brain,
                target_image_mask=thickness_results['BrainExtractionMask'],
                atlas_list=warped_images,
                label_list=warped_labels,
                dimension=3,
                num_threads=self.num_threads if self.num_threads > 1 else None
            )
            
            return {
                'segmentation': fusion['segmentation'],
                'intensity': fusion['intensity'],
                'probabilityimages': fusion['probabilityimages'],
                'segmentation_numbers': fusion['segmentation_numbers'],
                'BrainSegmentationN4': thickness_results['BrainSegmentationN4'],
                'BrainExtractionMask': thickness_results['BrainExtractionMask']
            }
            
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

    def _organize_bids_output(self, segmentation, bids_subject, bids_session=None):
        """Organize ANTs outputs in BIDS-compliant format.

        Parameters
        ----------
        segmentation : dict
            Segmentation results containing:
            - For Atropos: dictionary with 'segmentation' and 'probabilityimages'
            - For JointLabelFusion: dictionary with 'segmentation', 'intensity', 
              'probabilityimages', and 'segmentation_numbers'
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

        # Save labeled image
        label_filename = f"sub-{bids_subject}{session_part}_space-orig_dseg.nii.gz"
        label_path = anat_dir / label_filename
        ants.image_write(segmentation['segmentation'], str(label_path))
        
        # Calculate volumes and prepare stats
        labeled_image = segmentation['segmentation']
        labels = labeled_image.numpy()
        voxel_volume = np.prod(labeled_image.spacing)
        brain_volume = np.sum(labels > 0) * voxel_volume
        
        # Generate antslabelstats.csv
        labelstats_file = stats_dir / "antslabelstats.csv"
        with open(labelstats_file, 'w') as f:
            f.write("Label,Volume\n")  # Header required by NIDM converter
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label > 0:  # Skip background
                    volume = np.sum(labels == label) * voxel_volume
                    f.write(f"{int(label)},{volume:.2f}\n")
        
        # Generate antsbrainvols.csv
        brainvols_file = stats_dir / "antsbrainvols.csv"
        with open(brainvols_file, 'w') as f:
            f.write("Name,Value\n")  # Header required by NIDM converter
            f.write(f"BVOL,{brain_volume:.2f}\n")  # Brain volume required by NIDM converter
            
            # If we have tissue volumes from JointLabelFusion, add them
            if 'probabilityimages' in segmentation and len(segmentation['probabilityimages']) >= 3:
                # Calculate tissue volumes from probability maps
                csf_vol = np.sum(segmentation['probabilityimages'][0].numpy() > self.prob_threshold) * voxel_volume
                gm_vol = np.sum(segmentation['probabilityimages'][1].numpy() > self.prob_threshold) * voxel_volume
                wm_vol = np.sum(segmentation['probabilityimages'][2].numpy() > self.prob_threshold) * voxel_volume
                
                f.write(f"CSFVOL,{csf_vol:.2f}\n")
                f.write(f"GMVOL,{gm_vol:.2f}\n")
                f.write(f"WMVOL,{wm_vol:.2f}\n")
        
        # Save probability maps if available
        if 'probabilityimages' in segmentation:
            for idx, prob_img in enumerate(segmentation['probabilityimages']):
                prob_filename = f"sub-{bids_subject}{session_part}_space-orig_label-{idx+1}_probseg.nii.gz"
                prob_path = anat_dir / prob_filename
                ants.image_write(prob_img, str(prob_path))
        
        self.logger.info(f"Saved segmentation results for subject {bids_subject}")
        return {
            'brain_volume': brain_volume,
            'label_stats': str(labelstats_file),
            'brain_vols': str(brainvols_file),
            'segmentation': str(label_path)
        }

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
        
    def run_subject(self, subject_id, session_label=None, method='fusion'):
        """Run ANTs segmentation for a subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (with "sub-" prefix)
        session_label : str, optional
            Session label (with "ses-" prefix)
        method : str, optional
            Segmentation method ('quick' or 'fusion')
            
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
            
            # Load image
            try:
                img = ants.image_read(str(input_file))
                if img is None:
                    self.logger.error(f"Failed to load image {input_file}")
                    return False
            except Exception as e:
                self.logger.error(f"Failed to load image {input_file}: {str(e)}")
                return False
            
            # Run segmentation
            segmentation = self.segment_image(img, method=method)
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
        
    def compute_cortical_thickness(self, image, template_dir):
        """Compute cortical thickness using ANTs.
        
        Parameters
        ----------
        image : ants.ANTsImage
            Input brain image
        template_dir : str or Path
            Path to template directory containing required files
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'BrainSegmentationN4': N4 bias corrected image
            - 'BrainExtractionMask': Brain mask
            - 'TemplateToSubject1GenericAffine': Affine transform
            - 'TemplateToSubject0Warp': Warp transform
        """
        template_dir = Path(template_dir)
        
        # Load templates
        brain_template = ants.image_read(str(template_dir / 'T_template0.nii.gz'))
        reg_template = ants.image_read(str(template_dir / 'T_template0_BrainCerebellum.nii.gz'))
        prob_mask = ants.image_read(str(template_dir / 'T_template0_BrainCerebellumProbabilityMask.nii.gz'))
        ext_mask = ants.image_read(str(template_dir / 'T_template0_BrainCerebellumExtractionMask.nii.gz'))
        
        self.logger.info("Starting cortical thickness computation")
        
        # N4 bias field correction
        self.logger.info("Performing N4 bias field correction")
        n4_image = ants.n4_bias_field_correction(image)
        
        # Initial brain extraction using registration template and probability mask
        self.logger.info("Performing initial brain extraction")
        init_reg = ants.registration(
            fixed=reg_template,
            moving=n4_image,
            type_of_transform='Rigid'
        )
        
        # Apply transforms to probability mask
        init_mask = ants.apply_transforms(
            fixed=n4_image,
            moving=prob_mask,
            transformlist=init_reg['invtransforms'],
            interpolator='Linear'
        )
        
        # Create initial brain mask
        init_brain_mask = init_mask > 0.5
        init_brain = n4_image * init_brain_mask
        
        # Perform template registration with brain-extracted image
        self.logger.info("Registering to template")
        reg = ants.registration(
            fixed=brain_template,
            moving=init_brain,
            type_of_transform='SyN',
            mask=init_brain_mask
        )
        
        # Apply registration to extraction mask
        brain_mask = ants.apply_transforms(
            fixed=n4_image,
            moving=ext_mask,
            transformlist=reg['invtransforms'],
            interpolator='NearestNeighbor'
        )
        
        return {
            'BrainSegmentationN4': n4_image,
            'BrainExtractionMask': brain_mask,
            'TemplateToSubject1GenericAffine': reg['fwdtransforms'][0],
            'TemplateToSubject0Warp': reg['fwdtransforms'][1]
        }
        