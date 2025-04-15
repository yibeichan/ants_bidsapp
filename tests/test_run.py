import os
import tempfile
import shutil
import unittest
import logging
from unittest.mock import patch, MagicMock, PropertyMock
import sys
from pathlib import Path
import json
import pkg_resources
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Check if running in CI environment
IN_CI = os.environ.get('CI', 'false').lower() == 'true'

# Mock the external dependencies
mock_ants = MagicMock()

# Configure ANTs mock
def mock_image_read(*args, **kwargs):
    mock_img = MagicMock()
    mock_img.numpy.return_value = np.zeros((10, 10, 10))
    mock_img.spacing = [1.0, 1.0, 1.0]
    mock_img.__gt__ = lambda self, other: np.ones((10, 10, 10), dtype=bool)
    mock_img.__mul__ = lambda self, other: mock_img
    return mock_img

def mock_n4_correction(*args, **kwargs):
    mock_img = MagicMock()
    mock_img.numpy.return_value = np.zeros((10, 10, 10))
    mock_img.spacing = [1.0, 1.0, 1.0]
    mock_img.__gt__ = lambda self, other: np.ones((10, 10, 10), dtype=bool)
    mock_img.__mul__ = lambda self, other: mock_img
    return mock_img

def mock_registration(*args, **kwargs):
    return {
        'warpedmovout': mock_image_read(),
        'fwdtransforms': ['transform1', 'transform2'],
        'invtransforms': ['transform3', 'transform4']
    }

def mock_apply_transforms(*args, **kwargs):
    return mock_image_read()

def mock_get_mask(*args, **kwargs):
    return mock_image_read()

mock_ants.image_read = mock_image_read
mock_ants.n4_bias_field_correction = mock_n4_correction
mock_ants.registration = mock_registration
mock_ants.apply_transforms = mock_apply_transforms
mock_ants.get_mask = mock_get_mask
mock_ants.joint_label_fusion.return_value = {
    'segmentation': mock_image_read(),
    'intensity': mock_image_read(),
    'probabilityimages': [mock_image_read() for _ in range(3)],
    'segmentation_numbers': [1, 2, 3]
}

# Mock BIDSLayout
mock_layout = MagicMock()
mock_layout.get_subjects.return_value = ['01']
mock_layout.get_sessions.return_value = ['01']

# Configure the mocks
sys.modules['ants'] = mock_ants
sys.modules['numpy'] = np
sys.modules['nibabel'] = MagicMock()

# Patch BIDSLayout
with patch('bids.BIDSLayout') as MockBIDSLayout:
    MockBIDSLayout.return_value = mock_layout
    from src.run import process_participant, process_session, main, nidm_conversion, get_bids_version, create_dataset_description

class TestRun(unittest.TestCase):
    """Test cases for the run module"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.bids_dir = os.path.join(self.temp_dir, "bids")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # Create directories
        os.makedirs(self.bids_dir)
        os.makedirs(self.output_dir)
        
        # Create a simple BIDS structure
        self.create_bids_structure()
        
        # Set up logger
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)
    
    def tearDown(self):
        """Clean up test fixtures after each test"""
        shutil.rmtree(self.temp_dir)
    
    def create_bids_structure(self):
        """Create a minimal BIDS directory structure for testing"""
        # Create dataset_description.json
        os.makedirs(os.path.join(self.bids_dir, "sub-01", "anat"))
        with open(os.path.join(self.bids_dir, "dataset_description.json"), "w") as f:
            f.write('{"Name": "Test Dataset", "BIDSVersion": "1.7.0"}')
        
        # Create a dummy T1w file
        with open(os.path.join(self.bids_dir, "sub-01", "anat", 
                              "sub-01_T1w.nii.gz"), "w") as f:
            f.write("dummy data")

    def create_derivatives_structure(self):
        """Create a minimal derivatives structure for testing NIDM conversion"""
        derivatives_dir = os.path.join(self.output_dir, "ants_bidsapp", "ants-seg")
        subject_dir = os.path.join(derivatives_dir, "sub-01")
        anat_dir = os.path.join(subject_dir, "anat")
        stats_dir = os.path.join(subject_dir, "stats")
        
        os.makedirs(anat_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        
        # Create required files
        seg_file = os.path.join(anat_dir, "sub-01_space-orig_dseg.nii.gz")
        labelstats_file = os.path.join(stats_dir, "antslabelstats.csv")
        brainvols_file = os.path.join(stats_dir, "antsbrainvols.csv")
        
        # Write dummy content
        with open(seg_file, 'wb') as f:
            f.write(b'dummy segmentation')  # Write as binary
        
        with open(labelstats_file, 'w') as f:
            f.write("Label,Volume\n1,1000.0\n2,2000.0\n")
        
        with open(brainvols_file, 'w') as f:
            f.write("Name,Value\nBVOL,3000.0\nCSFVOL,1000.0\nGMVOL,1000.0\nWMVOL,1000.0\n")
        
        return derivatives_dir

    def test_nidm_conversion(self):
        """Test NIDM conversion with proper file structure"""
        # Create derivatives structure
        derivatives_dir = self.create_derivatives_structure()
        nidm_dir = os.path.join(self.output_dir, "ants_bidsapp", "nidm")
        os.makedirs(nidm_dir, exist_ok=True)
        
        # Run NIDM conversion
        with patch('subprocess.run') as mock_run:
            # Configure mock to return success
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = "Success"
            mock_process.stderr = ""
            mock_run.return_value = mock_process
            
            result = nidm_conversion(
                self.logger,
                derivatives_dir,
                nidm_dir,
                "01",
                verbose=True
            )
            
            # Check results
            self.assertTrue(result)
            self.assertTrue(os.path.exists(nidm_dir))
            
            # Verify subprocess call
            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            
            # Verify all paths in command are strings
            self.assertTrue(all(isinstance(arg, str) for arg in cmd_args))
            
            # Verify required arguments are present
            self.assertIn("-f", cmd_args)
            self.assertIn("-subjid", cmd_args)
            self.assertIn("-j", cmd_args)
            
            # Verify file paths exist
            paths_str = [arg for arg in cmd_args if ".csv" in arg or ".nii.gz" in arg][0]
            paths = paths_str.split(",")
            self.assertEqual(len(paths), 3)  # Should have 3 paths
            for path in paths:
                self.assertTrue(os.path.exists(path))

    def test_nidm_conversion_missing_files(self):
        """Test NIDM conversion with missing required files"""
        derivatives_dir = os.path.join(self.output_dir, "ants_bidsapp", "ants-seg")
        nidm_dir = os.path.join(self.output_dir, "ants_bidsapp", "nidm")
        os.makedirs(derivatives_dir)
        
        result = nidm_conversion(
            self.logger,
            derivatives_dir,
            nidm_dir,
            "01",
            verbose=True
        )
        
        self.assertFalse(result)

    @unittest.skipIf(IN_CI, "Skip in CI environment as it requires modifying template directory")
    def test_process_participant(self):
        """Test participant level processing with NIDM conversion"""
        class Args:
            def __init__(self, test_instance):
                self.bids_dir = test_instance.bids_dir
                self.output_dir = test_instance.output_dir
                self.participant_label = "01"
                self.session_label = None
                self.modality = "T1w"
                self.prob_threshold = 0.5
                self.method = "quick"
                self.skip_nidm = False
                self.num_threads = 1
                self.verbose = True
                self.skip_bids_validation = True
        
        args = Args(self)
        
        # Create template directory structure in the project root
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'data'
        template_dir = data_dir / 'OASIS-30_Atropos_template'
        os.makedirs(template_dir, exist_ok=True)
        
        # Create dummy template files
        for fname in ['T_template0.nii.gz', 'T_template0_BrainCerebellum.nii.gz',
                     'T_template0_BrainCerebellumProbabilityMask.nii.gz',
                     'T_template0_BrainCerebellumExtractionMask.nii.gz']:
            with open(template_dir / fname, 'w') as f:
                f.write('dummy template')
        
        # Create Priors directory
        priors_dir = template_dir / 'Priors2'
        os.makedirs(priors_dir, exist_ok=True)
        for i in range(1, 7):
            with open(priors_dir / f'priors{i}.nii.gz', 'w') as f:
                f.write('dummy prior')

        # Create a real numpy array for testing
        test_array = np.zeros((10, 10, 10))
        test_array[5:8, 5:8, 5:8] = 1  # Create some non-zero values
        
        # Create mock segmentation results
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = test_array
        mock_seg.spacing = [1.0, 1.0, 1.0]
        
        mock_prob = MagicMock()
        mock_prob.numpy.return_value = test_array
        mock_prob.spacing = [1.0, 1.0, 1.0]
        
        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': [mock_prob, mock_prob, mock_prob],
            'BrainSegmentationN4': mock_seg,
            'BrainExtractionMask': mock_seg
        }
        
        with patch('src.run.ANTsSegmentation') as MockSegmenter, \
             patch('src.run.nidm_conversion') as mock_nidm, \
             patch('bids.BIDSLayout') as MockBIDSLayout:
            
            # Configure BIDSLayout mock
            mock_layout = MagicMock()
            mock_layout.get_subjects.return_value = ['01']
            MockBIDSLayout.return_value = mock_layout
            
            # Create and configure the mock segmenter instance
            mock_instance = MockSegmenter.return_value
            
            # Configure the run_subject method
            def mock_run_subject(subject_id, session_label=None, method='fusion'):
                # Create output directories
                subject_dir = os.path.join(self.output_dir, 'ants_bidsapp', 'ants-seg', subject_id)
                anat_dir = os.path.join(subject_dir, 'anat')
                stats_dir = os.path.join(subject_dir, 'stats')
                os.makedirs(anat_dir, exist_ok=True)
                os.makedirs(stats_dir, exist_ok=True)
                
                # Create dummy stats files
                with open(os.path.join(stats_dir, 'antslabelstats.csv'), 'w') as f:
                    f.write("Label,Volume\n1,1000.0\n2,2000.0\n")
                with open(os.path.join(stats_dir, 'antsbrainvols.csv'), 'w') as f:
                    f.write("Name,Value\nBVOL,3000.0\n")
                
                # Create dummy segmentation file
                with open(os.path.join(anat_dir, f"{subject_id}_space-orig_dseg.nii.gz"), 'w') as f:
                    f.write("dummy segmentation")
                    
                return True
            
            mock_instance.run_subject.side_effect = mock_run_subject
            mock_instance.segment_image.return_value = segmentation_results
            
            # Configure NIDM mock
            mock_nidm.return_value = True
            
            # Run processing
            result = process_participant(args, self.logger)
            
            # Verify results
            self.assertEqual(result, 0)
            mock_instance.run_subject.assert_called_once()
            mock_nidm.assert_called_once()

    @unittest.skipIf(IN_CI, "Skip in CI environment as it requires modifying template directory")
    def test_process_session(self):
        """Test session level processing with NIDM conversion"""
        # Create session structure
        session_dir = os.path.join(self.bids_dir, "sub-01", "ses-01", "anat")
        os.makedirs(session_dir, exist_ok=True)
        with open(os.path.join(session_dir, "sub-01_ses-01_T1w.nii.gz"), "w") as f:
            f.write("dummy data")
        
        class Args:
            def __init__(self, test_instance):
                self.bids_dir = test_instance.bids_dir
                self.output_dir = test_instance.output_dir
                self.participant_label = "01"
                self.session_label = "01"
                self.modality = "T1w"
                self.prob_threshold = 0.5
                self.method = "quick"
                self.skip_nidm = False
                self.num_threads = 1
                self.verbose = True
                self.skip_bids_validation = True
        
        args = Args(self)
        
        # Create template directory structure in the project root
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'data'
        template_dir = data_dir / 'OASIS-30_Atropos_template'
        os.makedirs(template_dir, exist_ok=True)
        
        # Create dummy template files
        for fname in ['T_template0.nii.gz', 'T_template0_BrainCerebellum.nii.gz',
                     'T_template0_BrainCerebellumProbabilityMask.nii.gz',
                     'T_template0_BrainCerebellumExtractionMask.nii.gz']:
            with open(template_dir / fname, 'w') as f:
                f.write('dummy template')
        
        # Create Priors directory
        priors_dir = template_dir / 'Priors2'
        os.makedirs(priors_dir, exist_ok=True)
        for i in range(1, 7):
            with open(priors_dir / f'priors{i}.nii.gz', 'w') as f:
                f.write('dummy prior')

        # Create a real numpy array for testing
        test_array = np.zeros((10, 10, 10))
        test_array[5:8, 5:8, 5:8] = 1  # Create some non-zero values
        
        # Create mock segmentation results
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = test_array
        mock_seg.spacing = [1.0, 1.0, 1.0]
        
        mock_prob = MagicMock()
        mock_prob.numpy.return_value = test_array
        mock_prob.spacing = [1.0, 1.0, 1.0]
        
        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': [mock_prob, mock_prob, mock_prob],
            'BrainSegmentationN4': mock_seg,
            'BrainExtractionMask': mock_seg
        }
        
        with patch('src.run.ANTsSegmentation') as MockSegmenter, \
             patch('src.run.nidm_conversion') as mock_nidm, \
             patch('bids.BIDSLayout') as MockBIDSLayout:
            
            # Configure BIDSLayout mock
            mock_layout = MagicMock()
            mock_layout.get_subjects.return_value = ['01']
            mock_layout.get_sessions.return_value = ['01']
            MockBIDSLayout.return_value = mock_layout
            
            # Create and configure the mock segmenter instance
            mock_instance = MockSegmenter.return_value
            
            # Configure the run_subject method
            def mock_run_subject(subject_id, session_label=None, method='fusion'):
                # Create output directories
                subject_dir = os.path.join(self.output_dir, 'ants_bidsapp', 'ants-seg', subject_id)
                if session_label:
                    subject_dir = os.path.join(subject_dir, session_label)
                anat_dir = os.path.join(subject_dir, 'anat')
                stats_dir = os.path.join(subject_dir, 'stats')
                os.makedirs(anat_dir, exist_ok=True)
                os.makedirs(stats_dir, exist_ok=True)
                
                # Create dummy stats files
                with open(os.path.join(stats_dir, 'antslabelstats.csv'), 'w') as f:
                    f.write("Label,Volume\n1,1000.0\n2,2000.0\n")
                with open(os.path.join(stats_dir, 'antsbrainvols.csv'), 'w') as f:
                    f.write("Name,Value\nBVOL,3000.0\n")
                
                # Create dummy segmentation file
                seg_name = f"{subject_id}_ses-{session_label}_space-orig_dseg.nii.gz" if session_label else f"{subject_id}_space-orig_dseg.nii.gz"
                with open(os.path.join(anat_dir, seg_name), 'w') as f:
                    f.write("dummy segmentation")
                    
                return True
            
            mock_instance.run_subject.side_effect = mock_run_subject
            mock_instance.segment_image.return_value = segmentation_results
            
            # Configure NIDM mock
            mock_nidm.return_value = True
            
            # Run processing
            result = process_session(args, self.logger)
            
            # Verify results
            self.assertEqual(result, 0)
            mock_instance.run_subject.assert_called_once_with("sub-01", "ses-01", method="quick")
            mock_nidm.assert_called_once()

    def test_get_bids_version(self):
        """Test getting BIDS version from package"""
        # Test with mock package
        with patch('pkg_resources.get_distribution') as mock_dist:
            mock_dist.return_value = MagicMock(version="0.1.0")
            version = get_bids_version()
            self.assertEqual(version, "0.1.0")
            
        # Test fallback when package not found
        with patch('pkg_resources.get_distribution') as mock_dist:
            mock_dist.side_effect = pkg_resources.DistributionNotFound()
            version = get_bids_version()
            self.assertEqual(version, "1.8.0")
    
    def test_create_dataset_description(self):
        """Test creation of dataset_description.json"""
        test_dir = os.path.join(self.temp_dir, "test_desc")
        os.makedirs(test_dir)
        
        # Test with mock version
        with patch('src.run.get_bids_version') as mock_version:
            mock_version.return_value = "0.1.0"
            create_dataset_description(test_dir, "1.0.0")
            
            # Check if file exists
            desc_file = os.path.join(test_dir, "dataset_description.json")
            self.assertTrue(os.path.exists(desc_file))
            
            # Check content
            with open(desc_file) as f:
                desc = json.load(f)
                self.assertEqual(desc["BIDSVersion"], "0.1.0")
                self.assertEqual(desc["GeneratedBy"][0]["Version"], "1.0.0")

if __name__ == "__main__":
    unittest.main()
