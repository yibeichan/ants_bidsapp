import os
import tempfile
import shutil
import unittest
import logging
from unittest.mock import patch, MagicMock
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Mock the external dependencies
sys.modules['ants'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['nibabel'] = MagicMock()

from src.run import run_participant_level, main

class TestRunDirect(unittest.TestCase):
    """Direct test cases for the run module without using command line arguments"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # Create directories
        os.makedirs(self.input_dir)
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
        os.makedirs(os.path.join(self.input_dir, "sub-test_subject_0", "anat"))
        with open(os.path.join(self.input_dir, "dataset_description.json"), "w") as f:
            f.write('{"Name": "Test Dataset", "BIDSVersion": "1.7.0"}')
        
        # Create a dummy T1w file
        with open(os.path.join(self.input_dir, "sub-test_subject_0", "anat", 
                              "sub-test_subject_0_T1w.nii.gz"), "w") as f:
            f.write("dummy data")

    @patch('src.run.ANTsSegmentation')
    @patch('src.run.subprocess.run')
    def test_run_participant_direct(self, mock_subprocess, mock_segmenter):
        """Test running participant analysis directly without command line args"""
        # Set up mock
        mock_instance = MagicMock()
        mock_segmenter.return_value = mock_instance
        mock_instance.run_subject.return_value = True
        
        # Mock subprocess.run to avoid actual NIDM conversion
        mock_subprocess.return_value = MagicMock(returncode=0)
    
        # Create args object directly
        class Args:
            def __init__(self, test_instance):
                self.bids_dir = test_instance.input_dir
                self.output_dir = test_instance.output_dir
                self.participant_label = ["test_subject_0"]
                self.session_label = None
                self.modality = "T1w"
                self.prob_threshold = 0.5
                self.priors = None
                self.skip_nidm = False
                self.num_threads = 1
                self.verbose = True
    
        args = Args(self)
    
        # Run the function
        result = run_participant_level(args, self.logger)
    
        # Check results
        self.assertEqual(result, 0)
        mock_segmenter.assert_called_once()
        
        # Check that run_subject was called with the correct arguments
        # We need to check the call_args instead of using assert_called_once_with
        # because the actual call might not include the None parameter
        self.assertTrue(mock_instance.run_subject.called)
        call_args = mock_instance.run_subject.call_args
        self.assertEqual(call_args[0][0], "test_subject_0")
        # The second argument might be None or omitted, so we don't check it strictly

    @patch('src.run.run_participant_level')
    def test_main_direct(self, mock_run_participant):
        """Test main function directly without command line args"""
        # Set up mock
        mock_run_participant.return_value = 0

        # Run main with mocked functions
        with patch('src.run.parse_arguments') as mock_parse_args, \
             patch('src.run.setup_logger') as mock_logger:
            
            # Configure mocks
            mock_args = MagicMock()
            mock_args.analysis_level = "participant"
            mock_args.output_dir = self.output_dir
            mock_parse_args.return_value = mock_args
            mock_logger.return_value = self.logger

            # Run main
            result = main()

        # Check results
        self.assertEqual(result, 0)
        mock_run_participant.assert_called_once_with(mock_args, mock_logger.return_value)

if __name__ == "__main__":
    unittest.main()
