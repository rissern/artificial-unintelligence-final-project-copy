import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.functional import relu, pad, interpolate
import segmentation_models_pytorch as smp
from torchvision.transforms import Resize


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
        # Padding here would be better
        final_size = x.size()[2]
        remainder = x.size()[2] % 32

        if remainder != 0:
            final_size += (32 - remainder)
        # interpolation = Bilinear
        x = interpolate(x, size=(final_size, final_size), mode="bilinear")
        x = super().forward(x)
        # Crop out the padding
        x = self.pool(x)
        return x

        