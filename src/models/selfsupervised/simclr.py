# base code from: https://docs.lightly.ai/self-supervised-learning/examples/simsiam.html

import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from typing import Sequence
import segmentation_models_pytorch as smp


# code from: pytorch deeplabv3.py https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, atrous_rates: Sequence[int] = (12, 24, 36)) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

# code from: pytorch fcn.py https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py
class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)

class SimCLR(nn.Module):
    """
    SimCLR model for self-supervised learning with contrastive learning.

    Args:
        in_channels:
            The number of input channels.
        out_channels:
            The number of output channels
        scale_factor:
            number of input pixels that map to 1 output pixel,
            for example, if the input is 800x800 and the output is 16x6
            then the scale factor is 800/16 = 50.
        seg_model:
            The segmentation model architecture to use. Either 'resnet50' or 'deeplabv3'.
        out_dim:
            The dimensionality of the embedding.
        hidden_dim:
            The dimensionality of the hidden layer in the projection head
        mode:
            The mode of the model. Either 'pretrain' or 'finetune'.
    """
    def __init__(self, in_channels, out_channels, scale_factor=50, seg_model="resnet50", out_dim=2048, hidden_dim=2048, mode="pretrain",  **kwargs):
        super(SimCLR, self).__init__()

        # mode can be either 'pretrain' or 'finetune'
        assert mode in ["pretrain", "finetune"], "mode must be either 'pretrain' or 'finetune'"
        self.mode = mode
        self.seg_model = seg_model


        if seg_model == "resnet50" or seg_model == "deeplabv3":
            resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            resnet.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )


            # print(*list(resnet.children()))

            self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # Keep the feature map output

            backbone_output_dim = 2048

            self.projection_head = SimCLRProjectionHead(
                input_dim=backbone_output_dim,
                hidden_dim=hidden_dim,
                output_dim=out_dim,
            )


            if seg_model == "resnet50":
                self.segmentation_head = FCNHead(in_channels=backbone_output_dim, channels=out_channels)

            elif seg_model == "deeplabv3":
                self.segmentation_head = DeepLabHead(in_channels=backbone_output_dim, num_classes=out_channels)

        elif seg_model == "unet++":
            self.unetpp = smp.UnetPlusPlus(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=in_channels,                 
                classes=out_channels
            )

            self.backbone = self.unetpp.encoder
            backbone_output_dim = 2048

            self.projection_head = SimCLRProjectionHead(
                input_dim=backbone_output_dim,
                hidden_dim=hidden_dim,
                output_dim=out_dim,
            )

        self.criterion = NTXentLoss()
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool = nn.AvgPool2d(kernel_size=scale_factor)

    def set_mode(self, mode: str):
        """
        Set mode of model

        Arguments:
            mode: str
                of of ["pretrain", "finetune"]
        """
        assert mode in ["pretrain", "finetune"], "mode must be either 'pretrain' or 'finetune'"
        self.mode = mode

    def get_mode(self):
        return self.mode


    def forward(self, x):
        # based on the mode, the forward pass will return the output of the projection head or the segmentation head
        if self.seg_model == "resnet50":
            return self.forward_resnet50(x)
        elif self.seg_model == "deeplabv3":
            return self.forward_resnet50(x)
        elif self.seg_model == "unet++":
            return self.forward_unetpp(x)
        
    def forward_unetpp(self, x):
        """
        Forward pass for the UNet++ model
        """     

        input_shape = x.shape[-2:]          

        desired_size = (x.shape[-2]//32*32, x.shape[-1]//32*32)
        x = nn.functional.interpolate(x, size=desired_size, mode="bilinear", align_corners=False)

        if self.mode == "pretrain":
            features = self.backbone(x)[-1]
            x = self.adaptive_pool(features)
            x = x.flatten(start_dim=1)
            x = self.projection_head(x)
            return x
        
        elif self.mode == "finetune":
            x = self.unetpp(x)
            # down/up sample the output to the original input
            x = nn.functional.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            x = self.pool(x)
            return x
        
    def forward_resnet50(self, x):
        """
        Forward pass for the ResNet50 model
        """
        input_shape = x.shape[-2:]            

        features = self.backbone(x)

        if self.mode == "pretrain":
            x = self.adaptive_pool(features)
            x = x.flatten(start_dim=1)
            x = self.projection_head(x)
            return x
        
        elif self.mode == "finetune":

            x = self.segmentation_head(features)

            # upsample the output to the original input
            x = nn.functional.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

            x = self.pool(x)
            return x


    def training_step(self, batch, batch_index):
        
        if self.mode == "pretrain":
            return self.pretrain_step(batch, batch_index)["loss"]
        elif self.mode == "finetune":
            return self.finetune_step(batch, batch_index)["loss"]
            
    def pretrain_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        
        return {
            "loss": loss,
        }
    
    def finetune_step(self, batch, batch_index):
        imgs, masks = batch
        pred = self.forward(imgs)
        loss = nn.functional.cross_entropy(pred, masks)
        return {
            "loss": loss,
            "pred": pred,
        }




if __name__ == '__main__':

    # sanity check model
    model = SimCLR(in_channels=33, out_channels=4)

    x = torch.rand(21, 33, 100, 100)

    z = model(x)

    model.set_mode("finetune")

    z = model(x)
