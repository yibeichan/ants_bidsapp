import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import numpy and nibabel for test image creation
import numpy as np
import nibabel as nib

# Import from the ANTs BIDS app
from src.antspy.wrapper import ANTsSegmentation


class TestANTsSegmentation(unittest.TestCase):
    """Test cases for the ANTs segmentation wrapper"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.temp_path = os.path.join(self.temp_dir, "temp")
        
        # Create directories
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        
        # Create test image
        self.test_image_path = os.path.join(self.input_dir, "test_image.nii.gz")
        with open(self.test_image_path, "w") as f:
            f.write("dummy data")
        
        # Initialize segmenter
        self.segmenter = ANTsSegmentation(
            input_path=self.input_dir,
            output_path=self.output_dir,
            temp_path=self.temp_path,
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
        self.assertEqual(self.segmenter.input_path, self.input_dir)
        self.assertEqual(self.segmenter.output_path, self.output_dir)
        self.assertEqual(self.segmenter.temp_path, self.temp_path)
        self.assertEqual(self.segmenter.modality, 'T1w')
        self.assertEqual(self.segmenter.prob_threshold, 0.5)
        self.assertEqual(self.segmenter.num_threads, 1)
    
    @patch('ants.image_read')
    def test_load_image(self, mock_image_read):
        """Test loading an image"""
        # Set up mock
        mock_image = MagicMock()
        mock_image_read.return_value = mock_image
        
        # Call the method
        image_path = self.test_image_path
        result = self.segmenter.load_image(image_path)
        
        # Check results
        mock_image_read.assert_called_once_with(image_path)
        self.assertEqual(result, mock_image)
    
    @patch('ants.n4_bias_field_correction')
    @patch('ants.get_mask')
    @patch('ants.mask_image')
    def test_preprocess_image(self, mock_mask_image, mock_get_mask, mock_n4_correction):
        """Test preprocessing an image"""
        # Set up mocks
        mock_image = MagicMock()
        mock_corrected = MagicMock()
        mock_mask = MagicMock()
        mock_brain = MagicMock()
        
        mock_n4_correction.return_value = mock_corrected
        mock_get_mask.return_value = mock_mask
        mock_mask_image.return_value = mock_brain
        
        # Call the method
        result = self.segmenter.preprocess_image(mock_image)
        
        # Check results
        mock_n4_correction.assert_called_once_with(mock_image)
        mock_get_mask.assert_called_once_with(mock_corrected)
        mock_mask_image.assert_called_once_with(mock_corrected, mock_mask)
        self.assertEqual(result, mock_brain)
    
    @patch('ants.atropos')
    @patch('ants.get_mask')
    def test_segment_image(self, mock_get_mask, mock_atropos):
        """Test segmenting an image"""
        # Set up mocks
        mock_image = MagicMock()
        mock_mask = MagicMock()
        mock_segmentation = {'segmentation': MagicMock(), 'probabilityimages': [MagicMock()]}
        
        mock_get_mask.return_value = mock_mask
        mock_atropos.return_value = mock_segmentation
        
        # Call the method
        result = self.segmenter.segment_image(mock_image)
        
        # Check results
        mock_get_mask.assert_called_once_with(mock_image)
        mock_atropos.assert_called_once()
        self.assertEqual(result, mock_segmentation)
    
    @patch('ants.image_write')
    @patch('ants.from_numpy')
    @patch('numpy.prod')
    @patch('numpy.sum')
    @patch('numpy.unique')
    @patch('numpy.array')
    def test_save_results(self, mock_array, mock_unique, mock_sum, mock_prod, mock_from_numpy, mock_image_write):
        """Test saving segmentation results"""
        # Set up mocks
        mock_segmentation = {
            'segmentation': MagicMock(),
            'probabilityimages': [MagicMock(), MagicMock(), MagicMock()]
        }
    
        # Configure numpy mocks
        mock_prod.return_value = 1.0
        mock_sum.return_value = 100.0
        mock_unique.return_value = [0, 1, 2]
        
        # Configure from_numpy mock
        mock_from_numpy.return_value = MagicMock()
        
        # Configure array mock to return itself for comparison operations
        mock_array_instance = MagicMock()
        mock_array_instance.__gt__.return_value = mock_array_instance
        mock_array_instance.astype.return_value = mock_array_instance
        mock_array.return_value = mock_array_instance
    
        # Configure segmentation mock
        mock_segmentation['segmentation'].numpy.return_value = mock_array_instance
        mock_segmentation['segmentation'].spacing = [1, 1, 1]
    
        # Configure probability image mocks
        for prob in mock_segmentation['probabilityimages']:
            prob.numpy.return_value = mock_array_instance
            prob.origin = [0, 0, 0]
            prob.spacing = [1, 1, 1]
            prob.direction = np.eye(3)
    
        # Call the method
        self.segmenter.save_results(mock_segmentation, "test_subject")
    
        # Verify that image_write was called for each probability image, binary mask, and the segmentation
        # 3 probability images + 3 binary masks + 1 segmentation = 7 calls
        self.assertEqual(mock_image_write.call_count, 7)


if __name__ == "__main__":
    unittest.main() 