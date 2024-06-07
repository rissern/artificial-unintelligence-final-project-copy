import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as torchvision_transforms

sys.path.append(".")
from src.preprocessing.subtile import Subtile
from src.utilities import SatelliteType


class ESDDataset(Dataset):
    """
    A PyTorch Dataset class for creating batches in a DataLoader.

    Parameters:
        processed_dir: Path to the processed directory
        transform: Compose of the transformations for training
        satellite_type_list: List of satellite types
        slice_size: Slice size for the subtiles

    Methods:
        __init__(processed_dir, transform, satellite_type_list, slice_size) -> None:
            Initializes the dataset and populates self.subtile_dirs.

        __len__() -> int:
            Returns the number of subtiles in the dataset.

        __getitem__(idx) -> Tuple[np.ndarray, np.ndarray]:
            Retrieves a subtile at idx, aggregates time bands, applies transformations (self.transform),
            and returns processed input (X) and ground truth (y).

    """

    def __init__(
        self,
        processed_dir: Path,
        transform: torchvision_transforms.transforms.Compose,
        satellite_type_list: List[SatelliteType],
        slice_size: Tuple[int, int] = (4, 4),
    ) -> None:
        self.transform = transform
        self.satellite_type_list = satellite_type_list
        self.slice_size = slice_size

        # --- start here ---
        # initialize a list for the subtile_dirs
        self.subtile_dirs = []

        # iterate over the tiles in the processed_dir / subtiles
        for tile in processed_dir.glob("subtiles/*"):

            # iterate over the subtiles within the tile
            for subtile in tile.glob("*"):

                # append the subtile to the subtile_dirs list
                self.subtile_dirs.append(subtile)

    def __len__(self) -> int:
        """
        Returns number of subtiles in the dataset

        Output: int
            length: number of subtiles in the dataset
        """
        # --- start here ---
        return len(self.subtile_dirs)

    def __aggregate_time(self, array: np.ndarray):
        """
        Aggregates time dimension in order to
        feed it to the machine learning model.

        This function needs to be changed in the
        final project to better suit your needs.

        For homework 2, you will simply stack the time bands
        such that the output is shaped (time*bands, width, height),
        i.e., all the time bands are treated as a new band.

        Input:
            img: np.ndarray
                (time, bands, width, height) array
        Output:
            new_img: np.ndarray
                (time*bands, width, height) array
        """
        # merge the time and bands dimension (hint: use np.stack or np.reshape)

        # assert dtype is float32
        assert array.dtype == np.float32, f"array.dtype: {array.dtype} != np.float32"
        
        time, bands, width, height = array.shape
        new_img = array.reshape(time*bands, width, height)

        return new_img

    def __getitem__(self, idx):
        """
        Loads subtile by its directory at index idx, then
            - aggregates times
            - performs self.transform

        Input:
            idx: int
                index of subtile with respect to self.subtile_dirs

        Output:
            X: np.ndarray | torch.Tensor
                input data to ML model, of shape (time*bands, width, height)
            y: np.ndarray | torch.Tensor
                ground truth, of shape (1, width, height)
        """
        # create a subtile and load by directory. The directory to load from will be the subtile_dirs at idx.
        subtile = Subtile.load_subtile_by_dir(self.subtile_dirs[idx], self.satellite_type_list, self.slice_size)

        # get the data_array_list and ground truth from the subtile satellite list and ground truth
        data_array_list = subtile.satellite_list
        ground_truth = subtile.ground_truth

        # convert items to their .values form, we are stripping away the xarray so we can feed the data into the model
        data_array_list = [data_array.values for data_array in data_array_list]
        ground_truth = ground_truth.values

        # initalize a list to store X
        X = []

        # iterate over each array in the stripped array from above
        for array in data_array_list:

            # aggregate time and append the array to X
            X.append(self.__aggregate_time(array))


        # concatenate X
        X = np.concatenate(X, axis=0)

        X = np.nan_to_num(X)

        # set y to be the ground truth data array .values squeezed on the 0 and 1 axis
        y = ground_truth.squeeze(0).squeeze(0)

        self_supervised_transforms = False

        # if the transform is not none
        if self.transform is not None:

            # apply the transform to X and y and store the result
            transform_result = self.transform({"X": X, "y": y})

            if type(transform_result) == dict:
                # set X to be the result for X
                X = transform_result["X"]

                # set y to be the result for y
                y = transform_result["y"]
            else:
                # if there is a type error, set X and y to be the transform result
                X = torch.from_numpy(np.array([transform_result[0]["X"], transform_result[1]["X"]]))
                y = torch.from_numpy(np.array([transform_result[0]["y"]-1, transform_result[1]["y"]-1]))
                self_supervised_transforms = True

        # return X and y-1, labels go from 1-4, so y-1 makes them zero indexed
        if self_supervised_transforms:
            return X, y
        return X, y-1
