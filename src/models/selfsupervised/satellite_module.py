import torch
import pytorch_lightning as pl
from torch.optim import Adam, SGD, AdamW
from torch import nn
import torchmetrics

from src.models.selfsupervised.simclr import SimCLR


class ESDSelfSupervised(pl.LightningModule):
    def __init__(
        self,
        model_type,
        in_channels,
        out_channels,
        learning_rate=1e-3,
        model_params: dict = {},
    ):
        """
        Constructor for ESDSelfSupervised class.
        """
        # CALL THE CONSTRUCTOR OF THE PARENT CLASS
        super().__init__()

        # use self.save_hyperparameters to ensure that the module will load
        self.save_hyperparameters()

        # store in_channels and out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate

        # if the model type is segmentation_cnn, initalize a unet as self.model
        # if the model type is unet, initialize a unet as self.model
        # if the model type is fcn_resnet_transfer, initialize a fcn_resnet_transfer as self.model
        if model_type == "SimClr":
            self.model = SimCLR(in_channels, **model_params)
        else:
            raise Exception(f"model_type not found: {model_type}")

    def forward(self, X):
        """
        Forward pass of the model.
        """
        return self.model.forward(X)

    def training_step(self, batch, batch_idx):
        
        loss = self.model.training_step(batch, batch_idx)

        # return loss
        self.log("train_loss", loss)
        return loss


    def configure_optimizers(self):
        
        return self.model.configure_optimizers()
