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
            self.model = SimCLR(in_channels, out_channels, **model_params)
        else:
            raise Exception(f"model_type not found: {model_type}")
        
        self.train_accuracy_metrics = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=out_channels,
            average="macro",
            multidim_average="global",
        )  # not sure the parameters are correct
        self.eval_accuracy_metrics = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=out_channels,
            average="macro",
            multidim_average="global",
        )  # not sure the parameters are correct

        # f1 score
        self.train_f1_metrics = torchmetrics.F1Score(
            task="multiclass",
            num_classes=out_channels, 
        )
        self.eval_f1_metrics = torchmetrics.F1Score(
            task="multiclass",
            num_classes=out_channels, 
        )

    def forward(self, X):
        """
        Forward pass of the model.
        """
        return self.model(X)

    def training_step(self, batch, batch_idx):
        
        loss = self.model.training_step(batch, batch_idx)

        # return loss
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):

        if self.model.mode == "finetune":
            out = self.model.finetune_step(batch, batch_idx)
            loss = out["loss"]
            pred = out["pred"]
            acc = self.eval_accuracy_metrics(pred, batch[1])
            self.log("eval_accuracy", acc, on_epoch=True)

        else:
            loss = self.model.training_step(batch, batch_idx)
        
        self.log("val_loss", loss)

        return loss


    def configure_optimizers(self):
        
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

