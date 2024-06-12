# Import
from pathlib import Path
import sys

sys.path.append(".")
import numpy as np
import torch

import matplotlib.offsetbox as osb
import matplotlib.pyplot as plt
from matplotlib import rcParams as rcp
from matplotlib.offsetbox import (AnnotationBbox, OffsetImage)
from PIL import Image

from sklearn import random_projection
from src.models.selfsupervised.satellite_module import ESDSelfSupervised

from src.utilities import ESDConfig, SatelliteType
from src.esd_data.datamodule import ESDDataModule
from src.visualization import restitch_plot
from src.preprocessing.subtile import Subtile

ROOT = Path.cwd()

def create_embeddings(options: ESDConfig, model_path: Path):

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

    # load model from checkpoint and set to eval mode
    esdselfsup = ESDSelfSupervised.load_from_checkpoint(model_path)
    esdselfsup.eval()

    # get a list of all processed files
    all_processed_files = [
        tile for tile in (Path(options.processed_dir) / "Val" / "subtiles").iterdir() if tile.is_dir()
    ]

    embeddings = []
    color_images = []

    # get embeddings for all tiles
    for tile in all_processed_files:

        subtile = Subtile(
            satellite_list=[],
            ground_truth=[],
            slice_size=options.slice_size,
            parent_tile_id=tile.name,
        )

        subtile.restitch(options.processed_dir / "Val", dataModule.satellite_type_list)

        # get the embeddings
        for i in range(options.slice_size[0]):
            for j in range(options.slice_size[1]):
                X, _ = restitch_plot.retrieve_subtile_file(
                    i, j, processed_dir=options.processed_dir / "Val", parent_tile_id=tile.name, datamodule=dataModule
                )

                # get sentinel-2 image
                sentinel_2 = subtile.load_subtile(options.processed_dir / "Val", [SatelliteType.S2], i, j)[0]
                rgb_image = (sentinel_2.sel(band=["04", "03", "02"]).to_numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)

                # reshape the data to fit the model
                X = X.reshape((1, X.shape[-3], X.shape[-2], X.shape[-1]))

                # move the image to the accelerator
                X = X.float().to(options.accelerator)
                embedding = esdselfsup.model.backbone(X.float())[-1]
                embedding = esdselfsup.model.adaptive_pool(embedding).flatten(start_dim=1)
                
                embedding = embedding.detach()

                embeddings.append(embedding)
                color_images.append(rgb_image)

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()

    projection = random_projection.GaussianRandomProjection(n_components=2)
    embeddings_2d = projection.fit_transform(embeddings)

    # normalize the embeddings to fit in the [0, 1] square
    M = np.max(embeddings_2d, axis=0)
    m = np.min(embeddings_2d, axis=0)
    embeddings_2d = (embeddings_2d - m) / (M - m)

    return embeddings, embeddings_2d, color_images


def plot_embeddings(embeddings, color_images):

    fig = plt.figure()
    fig.suptitle("Scatter Plot of the Sentinel-2 VIIRS Dataset")
    ax = fig.add_subplot(1, 1, 1)
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1.0, 1.0]])
    iterator = [i for i in range(embeddings.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings[i] - shown_images) ** 2, 1)
        if np.min(dist) < 2e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings[i]]]
        shown_images_idx.append(i)

    # plot image overlays
    for idx in shown_images_idx:
        thumbnail_size = int(rcp["figure.figsize"][0] * 2.0)
        img = color_images[idx]
        img = Image.fromarray(img).resize((thumbnail_size, thumbnail_size))
        img = np.array(img)
        img_box = AnnotationBbox(
            OffsetImage(img, cmap=plt.cm.gray_r), 
            embeddings[idx], 
            pad=0.2
            )
        ax.add_artist(img_box)

    # set aspect ratio
    ratio = 1.0 / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable="box")

    plt.show()



def frame_np_image(image_np: np.ndarray, w: int = 5):
    """Returns an image as a numpy array with a black frame of width w."""
    ny, nx, _ = image_np.shape
    # create an empty image with padding for the frame
    framed_img = np.zeros((w + ny + w, w + nx + w, 3), dtype=np.uint8)
    #framed_img = framed_img.astype(np.uint8)
    # put the original image in the middle of the new one
    framed_img[w:-w, w:-w] = image_np
    return framed_img


def plot_nearest_neighbors_3x3(example_image_idx: int, i: int, embeddings, images_rgb):
    """Plots the example image and its eight nearest neighbors."""
    n_subplots = 9
    # initialize empty figure
    fig = plt.figure()
    fig.suptitle(f"Nearest Neighbor Plot {i + 1}")
    #
    # get distances to the cluster center
    # distances = embeddings - embeddings[example_idx]
    # distances = np.power(distances, 2).sum(-1).squeeze()
    #
    distances = np.linalg.norm(embeddings - embeddings[example_image_idx], axis=1)
    # sort indices by distance to the center
    nearest_neighbors = np.argsort(distances)[:n_subplots]
    # show images
    for plot_offset, plot_idx in enumerate(nearest_neighbors):
        ax = fig.add_subplot(3, 3, plot_offset + 1)
        # get the corresponding filename

        if plot_offset == 0:
            ax.set_title(f"Images")
            plt.imshow(frame_np_image(images_rgb[plot_idx]))
        else:
            plt.imshow(images_rgb[plot_idx])
        # let's disable the axis
        plt.axis("off")
    plt.show()



if __name__ == "__main__":

    options = ESDConfig()

    model_path = ROOT / "models" / ESDConfig.model_type / "epoch=29-val_loss=1.58-eval_accuracy=0.59.ckpt"

    emb, emb_2d, rgb_images = create_embeddings(options, model_path)

    plot_embeddings(emb_2d, rgb_images)

    for i in range(5):
        plot_nearest_neighbors_3x3(i, i, emb_2d, rgb_images)