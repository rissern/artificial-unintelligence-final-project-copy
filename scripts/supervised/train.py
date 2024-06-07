import sys
from argparse import ArgumentParser

from pathlib import Path
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig, PROJ_NAME

ROOT = Path.cwd()


def train(options: ESDConfig):
    # initialize wandb
    wandb.init(project=PROJ_NAME)
    # setup the wandb logger
    wandb_logger = pl.loggers.WandbLogger(project=PROJ_NAME)

    # initialize the datamodule
    dataModule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        batch_size=options.batch_size,
        seed=options.seed,
        selected_bands=options.selected_bands,
        slice_size=options.slice_size,
        num_workers=options.num_workers,
    )

    # prepare the data
    dataModule.prepare_data()

    # create a model params dict to initialize ESDSegmentation
    # note: different models have different parameters
    model_param = {
        "model_type": options.model_type,
        "in_channels": options.in_channels,
        "out_channels": options.out_channels,
        "learning_rate": options.learning_rate,
    }

    # initialize the ESDSegmentation model
    ESDSegmentation_model = ESDSegmentation(**model_param)

    # Use the following callbacks, they're provided for you,
    # but you may change some of the settings
    # ModelCheckpoint: saves intermediate results for the neural network
    # in case it crashes
    # LearningRateMonitor: logs the current learning rate on weights and biases
    # RichProgressBar: nicer looking progress bar (requires the rich package)
    # RichModelSummary: shows a summary of the model before training (requires rich)
    callbacks = [
        ModelCheckpoint(
            dirpath=ROOT / "models" / options.model_type,
            filename="{epoch}-{eval_f1:.2f}-{eval_accuracy:.2f}-{eval_loss:.2f}",
            save_top_k=1,
            save_last=True,
            verbose=True,
            monitor="eval_f1",
            mode="max",
            # every_n_train_steps=1000,
        ),
        LearningRateMonitor(),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]

    # initialize trainer, set accelerator, devices, number of nodes, logger
    # max epochs and callbacks
    trainer = pl.Trainer(
        accelerator=options.accelerator,
        devices=options.devices,
        logger=wandb_logger,
        max_epochs=options.max_epochs,
        callbacks=callbacks,
        precision="32-true",
    )

    # run trainer.fit
    trainer.fit(ESDSegmentation_model, dataModule)


if __name__ == "__main__":
    # load dataclass arguments from yml file

    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        help="The model to initialize.",
        default=config.model_type,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="The learning rate for training model",
        default=config.learning_rate,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Number of epochs to train for.",
        default=config.max_epochs,
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=config.in_channels,
        help="Number of input channels",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=config.out_channels,
        help="Number of output channels",
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Depth of the encoders (CNN only)",
        default=config.depth,
    )
    parser.add_argument(
        "--n_encoders",
        type=int,
        help="Number of encoders (Unet only)",
        default=config.n_encoders,
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        help="Embedding size of the neural network (CNN/Unet)",
        default=config.embedding_size,
    )
    parser.add_argument(
        "--pool_sizes",
        help="A comma separated list of pool_sizes (CNN only)",
        type=str,
        default=config.pool_sizes,
    )
    parser.add_argument(
        "--kernel_size",
        help="Kernel size of the convolutions",
        type=int,
        default=config.kernel_size,
    )
    parser.add_argument(
        "--num_workers", help="Number of workers", type=int, default=config.num_workers
    )

    parse_args = parser.parse_args()

    config = ESDConfig(**parse_args.__dict__)
    train(config)
