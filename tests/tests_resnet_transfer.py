import sys
import unittest

import torch
import torch.nn as nn

sys.path.append('.')
from src.models.supervised.resnet_transfer import FCNResnetTransfer

class TestFCNResnetTransfer(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Define model input parameters
        self.in_channels = 3
        self.out_channels = 2
        self.scale_factor = 50
        
        # Create an instance of FCNResnetTransfer
        self.model = FCNResnetTransfer(self.in_channels, self.out_channels, self.scale_factor)
    
    @classmethod
    def tearDownClass(self):
        del self.model

    def test_init_attributes_set_correctly(self):

        # Check if the input_channels and output_channels are set correctly
        self.assertEqual(self.model.in_channels, self.in_channels)
        self.assertEqual(self.model.out_channels, self.out_channels)

        # Check if the model is loaded correctly from torch hub
        self.assertIsInstance(self.model.model, nn.Module)

    def test_model_first_layer_modified_correctly(self):

        # Check if the first convolution layer is modified correctly
        self.assertIsInstance(self.model.model.backbone.conv1, nn.Conv2d)
        self.assertEqual(self.model.model.backbone.conv1.in_channels, self.in_channels)
        self.assertEqual(self.model.model.backbone.conv1.out_channels, 64)
        self.assertEqual(self.model.model.backbone.conv1.kernel_size, (7, 7))
        self.assertEqual(self.model.model.backbone.conv1.stride, (2, 2))
        self.assertEqual(self.model.model.backbone.conv1.padding, (3, 3))
        self.assertFalse(self.model.model.backbone.conv1.bias)

    def test_model_last_layer_modified_correctly(self):

        # Check if the last convolution layer is modified correctly
        self.assertIsInstance(self.model.model.classifier[-1], nn.Conv2d)
        self.assertEqual(self.model.model.classifier[-1].in_channels, 512)
        self.assertEqual(self.model.model.classifier[-1].out_channels, self.out_channels)
        self.assertEqual(self.model.model.classifier[-1].kernel_size, (1, 1))
        self.assertEqual(self.model.model.classifier[-1].stride, (1, 1))

    def test_pooling_layer_created_correctly(self):

        # Check if the pooling layer is created correctly
        self.assertIsInstance(self.model.pool, nn.AvgPool2d)
        self.assertEqual(self.model.pool.kernel_size, self.scale_factor)

    def test_forward_single_batch_size(self):
        
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

    def test_forward_multiple_batch_size(self):
        
        # Define forward input parameters
        batch_size = 5
        width = 800
        height = 800

        # Create a random input tensor
        sample = torch.randn(batch_size, self.in_channels, width, height)

        # Perform forward pass
        output = self.model(sample)

        # Check if the output shape is correct
        expected_output_shape = (batch_size, self.out_channels, width // self.scale_factor, height // self.scale_factor)
        self.assertEqual(output.shape, expected_output_shape)

    def test_forward_output_type(self):
            
        # Define forward input parameters
        batch_size = 1
        width = 800
        height = 800

        # Create a random input tensor
        sample = torch.randn(batch_size, self.in_channels, width, height)

        # Perform forward pass
        output = self.model(sample)

        # Check if the output type is correct
        self.assertIsInstance(output, torch.Tensor)


if __name__ == '__main__':
    unittest.main()