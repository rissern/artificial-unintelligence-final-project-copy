import torch
import pytorch_lightning as pl
from torch.optim import Adam, SGD, AdamW
from torch import nn
import torchmetrics

from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer
from src.models.supervised.ra_unet import RA_UNet
from src.models.supervised.unetPP import UNetPP





class ESDSegmentation(pl.LightningModule):
    def __init__(
        self,
        model_type,
        in_channels,
        out_channels,
        learning_rate=1e-3,
        model_params: dict = {},
    ):
        """
        Constructor for ESDSegmentation class.
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
        if model_type == "SegmentationCNN":
            self.model = SegmentationCNN(in_channels, out_channels, **model_params)
        elif model_type == "UNet":
            self.model = UNet(in_channels, out_channels, **model_params)
        elif model_type == "FCNResnetTransfer":
            self.model = FCNResnetTransfer(in_channels, out_channels, **model_params)
        elif model_type == "RA_UNet":
            self.model = RA_UNet(in_channels, out_channels, **model_params)
        elif model_type == "UNet++":
            self.model = UNetPP(in_channels, out_channels, **model_params)
           

        else:
            raise Exception(f"model_type not found: {model_type}")

        # initialize the accuracy metrics for the semantic segmentation task
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
        # evaluate self.model
        return self.model(X)

    def training_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch

        # evaluate batch
        eval = self(sat_img)

        # calculate cross entropy loss
        loss = nn.CrossEntropyLoss()(eval, mask)

        # return loss
        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy_metrics(eval, mask), on_epoch=True)
        self.log("train_f1", self.train_f1_metrics(eval, mask), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch

        # evaluate batch for validation
        eval = self(sat_img)

        # get the class with the highest probability
        loss = nn.CrossEntropyLoss()(eval, mask)
        eval = torch.argmax(eval, dim=1)

        # evaluate each accuracy metric and log it in wandb
        acc = self.eval_accuracy_metrics(eval, mask)
        self.log("eval_loss", loss)
        self.log("eval_accuracy", acc, on_epoch=True)
        self.log("eval_f1", self.eval_f1_metrics(eval, mask), on_epoch=True)

        # return validation loss
        return loss

    def configure_optimizers(self):
        # initialize optimizer
        # optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        # optimizer = SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)

        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.001)

        # return optimizer
        return optimizer
