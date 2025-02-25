"""Data-loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import thread_map


class WSIBagDataset(Dataset):
    """
    A PyTorch Dataset for whole slide image (WSI) feature bags.

    This dataset loads feature bags from disk on demand. Each item consists of a
    2D tensor of shape `(num_patches, num_features)`, representing extracted
    patch features from a WSI, along with a corresponding label.

    Parameters
    ----------
    feature_paths : Sequence[Path | str]
        A list of file paths to the feature tensors.
    labels : npt.NDArray
        A NumPy array containing labels corresponding to each feature bag.

    Attributes
    ----------
    feature_paths : list[Path | str]
        A list of file paths where feature tensors are stored.
    labels : np.ndarray
        A NumPy array storing labels for each WSI.
    """

    def __init__(
        self,
        feature_paths: Sequence[Path | str],
        labels: npt.NDArray,
    ):
        self.feature_paths = feature_paths
        self.labels = np.asarray(labels)

        assert len(labels) == len(feature_paths)

        print("Initialized a dataset:")
        print(f"    N feature paths: {len(feature_paths)}")
        print(f"    Shape of labels: {labels.shape}")

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.feature_paths[index]
        features: torch.Tensor = torch.load(path, map_location="cpu")
        assert isinstance(features, torch.Tensor)
        label = torch.tensor(self.labels[index])

        assert features.ndim == 2, f"Expected 2-dim tensor but got {features.ndim}-dim"
        assert label.ndim == 1, f"Expected 1-dim tensor but got {label.ndim}-dim"

        return features, label


class InMemoryWSIBagDataset(Dataset):
    """
    A PyTorch Dataset for whole slide image (WSI) feature bags, where all bags are preloaded into memory.

    This dataset preloads all feature bags into memory to speed up access. Each item consists of a
    2D tensor of shape `(num_patches, num_features)`, representing extracted WSI features, along
    with a corresponding label.

    Parameters
    ----------
    feature_paths : Sequence[Path | str]
        A list of file paths to the feature tensors.
    labels : array-like
        An array-like object containing labels corresponding to each feature bag. If
        using this object for classification tasks, pass in a 1D vector of labels, where
        each labels an integer greater than or equal to 0. If using this for regression
        tasks, pass in a 2D matrix of labels, where each row contains the regression
        label(s) for each whole slide image.
    task : str
        Specifies whether the dataset is for classification or regression. The only
        available options are "classification" and "regression".
    verbose : bool, optional
        If True, print dataset initialization details.

    Attributes
    ----------
    feature_paths : list[Path]
        A list of file paths where feature tensors are stored.
    labels : np.ndarray
        A NumPy array storing labels for each WSI.
    bags : list[torch.Tensor]
        A list of feature tensors loaded into memory.
    """

    def __init__(
        self,
        feature_paths: Sequence[Path | str],
        labels: npt.NDArray,
        task: Literal["classification", "regression"],
        verbose: bool = True,
    ):
        self.feature_paths = [Path(p) for p in feature_paths]
        self.labels = np.asarray(labels)
        self.task = task
        self.verbose = verbose

        if self.task not in {"classification", "regression"}:
            raise ValueError(f"Unknown task '{task}'.")

        if len(labels) != len(feature_paths):
            raise ValueError("Number of feature paths must equal the number of labels.")

        if self.task == "regression" and self.labels.ndim != 2:
            raise ValueError(f"Expected 2-dim tensor but got {self.labels.ndim}-dim")
        elif self.task == "classification" and self.labels.ndim != 1:
            raise ValueError(f"Expected 1-dim tensor but got {self.labels.ndim}-dim")

        for p in self.feature_paths:
            assert p.exists(), f"Path not found: {p}"

        if self.verbose:
            print("Initialized a dataset:")
            print(f"    N feature paths: {len(feature_paths)}")
            print(f"    Shape of labels: {labels.shape}")
            print("Loading bags into memory...")

        self.bags: list[torch.Tensor] = thread_map(
            lambda p: torch.load(p, map_location="cpu"),
            self.feature_paths,
            max_workers=10,
        )

        if self.verbose:
            print("Done loading bags into memory.")
            num_bytes = sum(t.element_size() * t.numel() for t in self.bags)
            print(f"    {num_bytes / 1e9:0.2f} gigabytes")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.bags[index]
        features = features.float()
        label = torch.tensor(self.labels[index])

        if features.ndim != 2:
            raise ValueError(f"Expected 2-dim tensor but got {features.ndim}-dim")

        if self.task == "regression" and self.labels.ndim != 1:
            raise ValueError(f"Expected 1-dim tensor but got {self.labels.ndim}-dim")
        elif self.task == "classification" and self.labels.ndim != 0:
            raise ValueError(f"Expected 0-dim tensor but got {self.labels.ndim}-dim")

        return features, label
