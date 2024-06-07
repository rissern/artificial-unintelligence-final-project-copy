# base code from: https://docs.lightly.ai/self-supervised-learning/examples/simsiam.html

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

class SimCLR(pl.LightningModule):
    def __init__(self, in_channels, out_dim=512, hidden_dim=128,  **kwargs):
        super().__init__()

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
            hidden_dim=2048,
            output_dim=2048,
        )

        self.criterion = NTXentLoss()


    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
