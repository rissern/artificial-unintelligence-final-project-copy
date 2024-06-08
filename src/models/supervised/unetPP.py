import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.functional import relu, pad
import segmentation_models_pytorch as smp


class UNetPP(smp.UnetPlusPlus):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 3,
                 embedding_size: int = 64, scale_factor: int = 52, **kwargs):

        super().__init__(
                    encoder_name="resnet50",
                    encoder_weights="imagenet",
                    in_channels=in_channels,                 
                    classes=out_channels)
        self.pool = nn.MaxPool2d(kernel_size = (scale_factor, scale_factor))
    
    def forward(self, x):
        x = super().forward(x)
        x = self.pool(x)
        return x

        