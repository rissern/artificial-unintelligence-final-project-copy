import sys
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule


def preprocess_for_dim_reduction(
    esd_datamodule: ESDDataModule,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the data for the dimensionality reduction

    Input:
        esd_datamodule: ESDDataModule
            datamodule to load the data from

    Output:
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

        y_flat: np.ndarray
            Flattened ground truth of shape (sample, 1)
    """
    # --- start here ---
    # create two lists to store the samples for the data (X) and gt (y)
    data = []
    gt = []
    
    # for each X, y in the train dataloader
    for X, y in esd_datamodule.train_dataloader():
    
        # append X to its list
        data.append(X)
        
        # append y to its list
        gt.append(y)
        

    # concatenate both lists
    # the X list will now have shape (sample, time, band, width, height)
    data = np.stack(data, axis=0)
    
    # the y list will now have shape (sample, 1, 1, width, height)
    gt = np.stack(gt, axis=0)



    # reshape the X list to the new shape (sample,time*band*width*height)
    sample, time, band, width, height = data.shape

    X_flat = data.reshape(data.shape[0], -1)

    assert X_flat.shape == (sample, time*band*width*height), f"X_flat shape is {X_flat.shape} and should be {(sample, time*band*width*height)}"
    
    # reshape the y list to the new shape (sample, 1)
    sample, _, _, _ = gt.shape

    y_flat = gt.reshape(gt.shape[0], -1)

    assert y_flat.shape == (sample, 1), f"y_flat shape is {y_flat.shape} and should be {(sample, 1)}"
    

    # return the reshaped X and y
    return X_flat, y_flat


def perform_PCA(X_flat: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    Perform PCA on the input data

    Input:
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

    Output:
        X_pca: np.ndarray
            Dimensionality reduced array of shape (sample, n_components)
        pca: PCA
            PCA object
    """
    
    # create a PCA object with the n_components
    pca = PCA(n_components=n_components)
    
    # fit the PCA object to the X_flat and transform it
    X_pca = pca.fit_transform(X_flat)
    
    return X_pca, pca


def perform_TSNE(X_flat: np.ndarray, n_components: int) -> Tuple[np.ndarray, TSNE]:
    """
    Perform TSNE on the input data

    Input:
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

    Output:
        X_pca: np.ndarray
            Dimensionality reduced array of shape (sample, n_components)
        tsne: TSNE
            TSNE object
    """
    
    # create a TSNE object with the n_components
    tsne = TSNE(n_components=n_components)
    
    # fit the TSNE object to the X_flat and transform it
    X_tsne = tsne.fit_transform(X_flat)
    
    return X_tsne, tsne


def perform_UMAP(X_flat: np.ndarray, n_components: int) -> Tuple[np.ndarray, UMAP]:
    """
    Perform UMAP on the input data

    Input:
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

    Output:
        X_pca: np.ndarray
            Dimensionality reduced array of shape (sample, n_components)
        umap: UMAP
            UMAP object
    """
    
    # create a UMAP object with the n_components
    umap = UMAP(n_components=n_components)
    
    # fit the UMAP object to the X_flat
    umap.fit(X_flat)
    
    # transform the X_flat with the UMAP object
    X_umap = umap.transform(X_flat)
    
    return X_umap, umap