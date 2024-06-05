import sys
import unittest

import torch
import torch.nn as nn

sys.path.append('.')
from src.models.supervised.resnet_transfer import FCNResnetTransfer

from src.models.supervised.ra_unet import RA_UNet as unet

class UNet(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Define model input parameters
        self.in_channels = 2
        self.out_channels = 5
        self.n_encoders = 2
        self.embedding_size = 64
        self.scale_factor = 50
        
        # Create an instance of Unet

        self.model = unet(self.in_channels, self.out_channels, embedding_size=self.embedding_size, scale_factor=self.scale_factor)
    
    @classmethod
    def tearDownClass(self):
        del self.model

    def test_init_attributes_set_correctly(self):

        # Check if the input_channels and output_channels are set correctly
        self.assertEqual(self.in_channels, self.in_channels)
        self.assertEqual(self.out_channels, self.out_channels)

        # Check if the model is loaded correctly from torch hub
        # self.assertIsInstance(self.model.model, nn.Module)
    def test_forward_default(self):
        
        # Define forward input parameters
        batch_size = 1
        width = 800
        height = 800

        # Create a random input tensor
        sample = torch.randn(batch_size, self.in_channels, width, height)

        # Perform forward pass
        output = self.model(sample)

        # Check if the output shape is correct
        expected_output_shape = (batch_size, self.out_channels, width // self.scale_factor, height // self.scale_factor)
        self.assertEqual(output.shape, expected_output_shape)
        
    def test_forward_single_batch_size(self):
        
        # Define forward input parameters
        batch_size = 1
        width = 200
        height = 200

        # Create a random input tensor
        sample = torch.randn(batch_size, self.in_channels, width, height)

        # Perform forward pass
        output = self.model(sample)

        # Check if the output shape is correct
        expected_output_shape = (batch_size, self.out_channels, width // self.scale_factor, height // self.scale_factor)
        self.assertEqual(output.shape, expected_output_shape)


    


if __name__ == '__main__':
    unittest.main()