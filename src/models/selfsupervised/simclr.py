# base code from: https://docs.lightly.ai/self-supervised-learning/examples/simsiam.html

import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead

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

        resnet = torchvision.models.resnet18()
        resnet.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection_head = SimCLRProjectionHead(
            input_dim=512,
            hidden_dim=hidden_dim,
            output_dim=out_dim,
        )

        self.criterion = NTXentLoss()

        if seg_model == "resnet50":
            self.segmentation_model = torchvision.models.segmentation.fcn_resnet50(
                num_classes=out_channels,
            )

            # change the input layer to accept the number of input channels
            self.segmentation_model.backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

        elif seg_model == "deeplabv3":
            self.segmentation_model = torchvision.models.segmentation.deeplabv3_resnet50(
                num_classes=out_channels,
            )

            # change the input layer to accept the number of input channels
            self.segmentation_model.backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        
        self.pool = nn.AvgPool2d(kernel_size=scale_factor)

    def set_mode(self, mode: str):
        assert mode in ["pretrain", "finetune"], "mode must be either 'pretrain' or 'finetune'"
        self.mode = mode

    def get_mode(self):
        return self.mode


    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)

        if self.mode == "pretrain":
            z = self.projection_head(features)
            # return nn.functional.normalize(z, dim=-1)
            return z
        
        elif self.mode == "finetune":
            z = self.segmentation_model(x)["out"]
            z = self.pool(z)
            return z

    def training_step(self, batch, batch_index):
        
        if self.mode == "pretrain":
            return self.pretrain_step(batch, batch_index)
        elif self.mode == "finetune":
            return self.finetune_step(batch, batch_index)
            
    def pretrain_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss
    
    def finetune_step(self, batch, batch_index):
        imgs, masks = batch
        pred = self.forward(imgs)
        loss = nn.functional.cross_entropy(pred, masks)
        return loss


