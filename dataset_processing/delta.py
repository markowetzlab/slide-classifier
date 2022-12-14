# Marcel Gehrung
#To be changed

import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
from tqdm import tqdm

from .image import IMG_EXTENSIONS, default_loader, make_dataset

class WholeSlideImageDataset(Dataset):
    """WholeSlideImage dataset."""

    def __init__(self, slideClass, foregroundOnly=False, transform=None):
        self.slideClass = slideClass
        self.foregroundOnly = foregroundOnly
        self.transform = transform

    def __len__(self):
        return self.slideClass.getTileCount(foregroundOnly=self.foregroundOnly)

    def __getitem__(self, idx):
        tileAddress = self.slideClass.ind2sub(idx, foregroundOnly=self.foregroundOnly)
        img = Image.fromarray(self.slideClass.getTile(
            tileAddress, writeToNumpy=True)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'tileAddress': tileAddress}

        return sample

# Taken from an existing open-source project:
# https://github.com/ufoym/imbalanced-dataset-sampler
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None, pre_weights=None, pre_weights_filename=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        # distribution of classes in the dataset
        label_to_count = {}
        print('Parsing labels for imbalanced data sampling...')
        if pre_weights is None:
            for idx in tqdm(self.indices):
                label = self._get_label(dataset, idx)
                if label in label_to_count:
                    label_to_count[label] += 1
                else:
                    label_to_count[label] = 1
            # weight for each sample
            weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                       for idx in self.indices]
            self.weights = torch.DoubleTensor(weights)
            if pre_weights_filename is not None:
                pickle.dump(self.weights, open(pre_weights_filename, 'wb'))
        else:
            self.weights = pre_weights

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        elif dataset_type is torch.utils.data.dataset.ConcatDataset:
            return dataset.__getitem__(idx)[1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class EmptyDirAcceptingImageFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
