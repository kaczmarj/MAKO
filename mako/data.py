"""Data-loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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
    A PyTorch Dataset for whole slide image (WSI) feature bags, where all data is preloaded into memory.

    This dataset preloads all feature bags into memory to speed up access. Each item consists of a
    2D tensor of shape `(num_patches, num_features)`, representing extracted WSI features, along
    with a corresponding label.

    Parameters
    ----------
    feature_paths : Sequence[Path | str]
        A list of file paths to the feature tensors.
    labels : npt.NDArray
        A NumPy array containing labels corresponding to each feature bag.

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
    ):
        self.feature_paths = [Path(p) for p in feature_paths]
        self.labels = np.asarray(labels)
        assert len(labels) == len(feature_paths)

        for p in self.feature_paths:
            assert p.exists(), f"Path not found: {p}"

        print("Initialized a dataset:")
        print(f"    N feature paths: {len(feature_paths)}")
        print(f"    Shape of labels: {labels.shape}")

        print("Loading bags into memory...")
        self.bags: list[torch.Tensor] = thread_map(
            lambda p: torch.load(p, map_location="cpu"),
            self.feature_paths,
            max_workers=10,
        )
        print("Done loading bags into memory.")
        num_bytes = sum(t.element_size() * t.numel() for t in self.bags)
        print(f"    {num_bytes / 1e9:0.2f} gigabytes")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.bags[index]
        features = features.float()
        label = torch.tensor(self.labels[index])

        assert features.ndim == 2, f"Expected 2-dim tensor but got {features.ndim}-dim"
        assert label.ndim == 1, f"Expected 1-dim tensor but got {label.ndim}-dim"

        return features, label
