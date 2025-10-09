import csv
import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Check if running in CI environment
IN_CI = os.environ.get('CI', 'false').lower() == 'true'

# Import numpy and nibabel for test image creation
import numpy as np
import nibabel as nib

# Mock ants module
mock_ants = MagicMock()
sys.modules['ants'] = mock_ants

# Configure ANTs mock
def mock_image_read(*args, **kwargs):
    mock_img = MagicMock()
    mock_img.numpy.return_value = np.zeros((10, 10, 10))
    mock_img.spacing = [1.0, 1.0, 1.0]
    mock_img.__gt__ = lambda self, other: np.ones((10, 10, 10), dtype=bool)
    mock_img.__mul__ = lambda self, other: mock_img
    return mock_img

mock_ants.image_read = mock_image_read
mock_ants.n4_bias_field_correction = mock_image_read
mock_ants.get_mask = mock_image_read
mock_ants.mask_image = mock_image_read

# Import after mocking
from src.antspy.wrapper import ANTsSegmentation

@unittest.skipIf(IN_CI, "Skip in CI environment as it requires template directory")
class TestANTsSegmentation(unittest.TestCase):
    """Test cases for the ANTs segmentation wrapper"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.bids_dir = self.temp_dir / "bids"
        self.output_dir = self.temp_dir / "derivatives"
        self.temp_path = self.temp_dir / "tmp"
        
        # Create directories
        self.bids_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        
        # Mock _load_templates method
        with patch('src.antspy.wrapper.ANTsSegmentation._load_templates') as mock_load:
            mock_load.return_value = {
                'T_template': mock_image_read(),
                'BrainCerebellum': mock_image_read(),
                'ProbabilityMask': mock_image_read(),
                'ExtractionMask': mock_image_read(),
                'priors': [mock_image_read() for _ in range(6)]
            }
            
            # Initialize segmenter
            self.segmenter = ANTsSegmentation(
                bids_dir=str(self.bids_dir),
                output_dir=str(self.output_dir),
                temp_dir=str(self.temp_path),
                modality='T1w',
                prob_threshold=0.5,
                num_threads=1,
                verbose=True
            )
    
    def tearDown(self):
        """Clean up test fixtures after each test"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization of the ANTsSegmentation class"""
        # Check that the instance was created correctly
        self.assertEqual(str(self.segmenter.bids_dir), str(self.bids_dir))
        self.assertEqual(str(self.segmenter.output_dir), str(self.output_dir))
        self.assertEqual(str(self.segmenter.temp_dir), str(self.temp_path))
        self.assertEqual(self.segmenter.modality, 'T1w')
        self.assertEqual(self.segmenter.prob_threshold, 0.5)
        self.assertEqual(self.segmenter.num_threads, 1)
    
    def test_load_image(self):
        """Test loading an image"""
        # Create a test image file
        test_file = self.temp_dir / "test.nii.gz"
        test_file.write_text("dummy data")
            
        # Test loading the image
        result = self.segmenter.load_image(str(test_file))
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'numpy'))
        self.assertTrue(hasattr(result, 'spacing'))
    
    def test_preprocess_image(self):
        """Test preprocessing an image"""
        # Create a mock input image
        mock_image = mock_image_read()
        
        # Test preprocessing
        result = self.segmenter.preprocess_image(mock_image)
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'numpy'))
        self.assertTrue(hasattr(result, 'spacing'))
    
    def test_organize_bids_output_single_session(self):
        """Test organizing outputs in BIDS format for single session"""
        # Create mock segmentation results
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = np.array([0, 1, 1, 2, 2, 2])  # 2 voxels of label 1, 3 voxels of label 2
        mock_seg.spacing = [1.0, 1.0, 1.0]  # 1mm isotropic

        # Create mock probability images
        mock_prob1 = MagicMock()
        mock_prob1.numpy.return_value = np.array([0.8, 0.2, 0.1, 0.1, 0.1, 0.1])
        mock_prob1.spacing = [1.0, 1.0, 1.0]

        mock_prob2 = MagicMock()
        mock_prob2.numpy.return_value = np.array([0.1, 0.7, 0.8, 0.1, 0.1, 0.1])
        mock_prob2.spacing = [1.0, 1.0, 1.0]

        mock_prob3 = MagicMock()
        mock_prob3.numpy.return_value = np.array([0.1, 0.1, 0.1, 0.8, 0.8, 0.8])
        mock_prob3.spacing = [1.0, 1.0, 1.0]

        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': [mock_prob1, mock_prob2, mock_prob3]
        }

        with patch('ants.image_write') as mock_write:
            # Call the method
            result = self.segmenter._organize_bids_output(segmentation_results, "01")

            # Check directory structure
            subject_dir = self.output_dir / "sub-01"
            anat_dir = subject_dir / "anat"
            stats_dir = subject_dir / "stats"
            
            self.assertTrue(anat_dir.exists())
            self.assertTrue(stats_dir.exists())

            # Check stats files exist and have correct format
            labelstats_file = stats_dir / "antslabelstats.csv"
            brainvols_file = stats_dir / "antsbrainvols.csv"
            
            self.assertTrue(labelstats_file.exists())
            self.assertTrue(brainvols_file.exists())

            # Verify labelstats.csv content
            with open(labelstats_file, newline='') as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["Label", "VolumeInVoxels", "Volume_mm3"])
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                volumes = {row["Label"]: int(float(row["VolumeInVoxels"])) for row in rows}
                self.assertEqual(volumes["1"], 2)
                self.assertEqual(volumes["2"], 3)

            # Verify brainvols.csv content
            with open(brainvols_file, newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                self.assertTrue({"BVOL", "CSFVOL", "GMVOL", "WMVOL"}.issubset(set(fieldnames)))
                rows = list(reader)
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertIn("BVOL", row)
                self.assertIn("CSFVOL", row)
                self.assertIn("GMVOL", row)
                self.assertIn("WMVOL", row)

            # Check that image_write was called for all images
            self.assertEqual(mock_write.call_count, 4)  # 1 segmentation + 3 probability maps

    def test_organize_bids_output_with_session(self):
        """Test organizing outputs in BIDS format with session"""
        # Create mock segmentation with minimal data
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = np.array([0, 1, 2])
        mock_seg.spacing = [1.0, 1.0, 1.0]

        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': []
        }

        with patch('ants.image_write') as mock_write:
            # Call the method with session
            result = self.segmenter._organize_bids_output(segmentation_results, "01", "01")

            # Check session directory structure
            session_dir = self.output_dir / "sub-01" / "ses-01"
            anat_dir = session_dir / "anat"
            stats_dir = session_dir / "stats"
            
            self.assertTrue(anat_dir.exists())
            self.assertTrue(stats_dir.exists())

            # Check segmentation file naming
            seg_file = anat_dir / "sub-01_ses-01_space-orig_dseg.nii.gz"
            self.assertTrue(mock_write.call_args_list[0][0][1].endswith(str(seg_file)))

            labelstats_file = stats_dir / "antslabelstats.csv"
            brainvols_file = stats_dir / "antsbrainvols.csv"

            self.assertTrue(labelstats_file.exists())
            self.assertTrue(brainvols_file.exists())

            with open(labelstats_file, newline='') as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, ["Label", "VolumeInVoxels", "Volume_mm3"])

            with open(brainvols_file, newline='') as f:
                reader = csv.DictReader(f)
                self.assertIn("BVOL", reader.fieldnames)

    def test_save_results(self):
        """Test save_results method with output directory handling"""
        # Create mock segmentation
        mock_seg = MagicMock()
        mock_seg.numpy.return_value = np.array([0, 1, 2])
        mock_seg.spacing = [1.0, 1.0, 1.0]

        segmentation_results = {
            'segmentation': mock_seg,
            'probabilityimages': []
        }

        # Test with default output directory
        with patch('ants.image_write'):
            result = self.segmenter.save_results(segmentation_results, "01")
            self.assertIn('brain_volume', result)
            self.assertIn('label_stats', result)
            self.assertIn('brain_vols', result)
            self.assertIn('segmentation', result)

        # Test with custom output directory
        custom_output = self.temp_dir / "custom_output"
        custom_output.mkdir(parents=True, exist_ok=True)
        
        with patch('ants.image_write'):
            result = self.segmenter.save_results(segmentation_results, "01", output_dir=str(custom_output))
            self.assertTrue(all(str(custom_output) in str(path) for path in result.values() if isinstance(path, str)))

    def test_run_subject(self):
        """Test run_subject method with BIDS structure"""
        # Create BIDS directory structure
        subject_dir = self.bids_dir / "sub-01" / "anat"
        subject_dir.mkdir(parents=True, exist_ok=True)
        t1_file = subject_dir / "sub-01_T1w.nii.gz"
        t1_file.write_text("dummy data")

        # Mock the necessary methods
        with patch('ants.image_read') as mock_read, \
             patch.object(self.segmenter, 'segment_image') as mock_segment, \
             patch.object(self.segmenter, 'save_results') as mock_save:

            # Configure mocks
            mock_read.return_value = MagicMock()
            mock_segment.return_value = {
                'segmentation': MagicMock(),
                'probabilityimages': []
            }
            mock_save.return_value = {
                'brain_volume': 1000.0,
                'label_stats': 'path/to/stats.csv',
                'brain_vols': 'path/to/vols.csv',
                'segmentation': 'path/to/seg.nii.gz'
            }

            # Run the method
            result = self.segmenter.run_subject("sub-01")

            # Verify the results
            self.assertTrue(result)
            mock_read.assert_called_once()
            mock_segment.assert_called_once()
            mock_save.assert_called_once()

if __name__ == "__main__":
    unittest.main() 
