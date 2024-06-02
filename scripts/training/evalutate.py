import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig
from src.visualization.restitch_plot import restitch_and_plot


def main(options):
    # initialize datamodule
    dataModule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        batch_size=options.batch_size,
        seed=options.seed,
        selected_bands=options.selected_bands,
        slice_size=options.slice_size,
    )

    # prepare data
    dataModule.prepare_data()
    dataModule.setup("fit")

    # load model from checkpoint
    # set model to eval mode
    model = ESDSegmentation.load_from_checkpoint(
        options.model_path
    )
    model.eval()

    # get a list of all processed tiles
    all_processed_tiles = [
        tile for tile in (Path(options.processed_dir) / "Val" / "subtiles").iterdir() if tile.is_dir()
    ]
    
    # for each tile
    for tile in all_processed_tiles:
        # run restitch and plot
        restitch_and_plot(
            options,
            datamodule=dataModule,
            model=model,
            parent_tile_id=tile.name,
            accelerator=options.accelerator,
            results_dir=options.results_dir,
        )

    


if __name__ == "__main__":
    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, help="Model path.", default=config.model_path
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.results_dir, help="Results dir"
    )
    main(ESDConfig(**parser.parse_args().__dict__))
