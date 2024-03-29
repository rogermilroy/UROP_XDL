import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import *
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import *


def create_split_loaders(dataset, batch_size, seed,
                         p_val=0.1, p_test=0.2, shuffle=True, extras={}, subset_size=0):
    """ Creates the DataLoader objects for the training, validation, and test sets.

    Params:
    -------
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - transform: A torchvision.transforms object - transformations to apply to each image
                 (Can be "transforms.Compose([transforms])")
    - p_val: (float) Percent (as decimal) of dataset to use for validation
    - p_test: (float) Percent (as decimal) of the dataset to split for testing
    - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
    - show_sample: (bool) Plot a mini-example as a grid of the dataset
    - extras: (dict)
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """

    # Dimensions and indices of training set
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)

    # if we want just a subset we do it after shuffling to ensure reasonable distribution.
    if subset_size > 0:
        dataset_size = subset_size
        all_indices = all_indices[:dataset_size]

    # Create the test split from the full dataset
    test_split = int(np.floor(p_test * dataset_size))
    train_ind, test_ind = all_indices[test_split:], all_indices[: test_split]

    # Separate a test split from the training dataset
    val_split = int(np.floor(p_val * len(train_ind)))
    train_ind, val_ind = train_ind[val_split:], train_ind[: val_split]

    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SubsetRandomSampler(train_ind)
    sample_test = SubsetRandomSampler(test_ind)
    sample_val = SubsetRandomSampler(val_ind)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sample_train, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=sample_test, num_workers=num_workers,
                             pin_memory=pin_memory)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers,
                            pin_memory=pin_memory)

    # Return the training, validation, test DataLoader objects
    return train_loader, test_loader, val_loader


class ChestXrayDataset(Dataset):
    """Custom Dataset class for the Chest X-Ray Dataset.

    The expected dataset is stored in the "/datasets/ChestXray-NIHCC/" on ieng6
    """

    def __init__(self, transform=transforms.ToTensor(), color='L'):
        """
        Args:
        -----
        - transform: A torchvision.transforms object -
                     transformations to apply to each image
                     (Can be "transforms.Compose([transforms])")
        - color: Specifies image-color format to convert to
                 (default is L: 8-bit pixels, black and white)

        Attributes:
        -----------
        - image_dir: The absolute filepath to the dataset on ieng6
        - image_info: A Pandas DataFrame of the dataset metadata
        - image_filenames: An array of indices corresponding to the images
        - labels: An array of labels corresponding to the each sample
        - classes: A dictionary mapping each disease name to an int between [0, 13]
        """

        self.transform = transform
        self.color = color
        self.image_dir = "../../ChestXrayImgs/images/"
        self.image_info = pd.read_csv("../../ChestXrayImgs/Data_Entry_2017.csv")
        self.image_filenames = self.image_info["Image Index"]
        self.labels = self.image_info["Finding Labels"]
        self.classes = {0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion",
                        3: "Infiltration", 4: "Mass", 5: "Nodule", 6: "Pneumonia",
                        7: "Pneumothorax", 8: "Consolidation", 9: "Edema",
                        10: "Emphysema", 11: "Fibrosis",
                        12: "Pleural_Thickening", 13: "Hernia"}

    def __len__(self):

        # Return the total number of data samples
        return len(self.image_filenames)

    def __getitem__(self, ind):
        """Returns the image and its label at the index 'ind'
        (after applying transformations to the image, if specified).

        Params:
        -------
        - ind: (int) The index of the image to get

        Returns:
        --------
        - A tuple (image, label)
        """

        # Compose the path to the image file from the image_dir + image_name
        image_path = os.path.join(self.image_dir, self.image_filenames.ix[ind])

        # Load the image
        image = Image.open(image_path).convert(mode=str(self.color))

        # Verify that image is in Tensor format
        if type(image) is not torch.Tensor:
            image = ToTensor(image)

        # If a transform is specified, apply it
        if self.transform is not None:
            image = self.transform(image)

        # Convert multi-class label into binary encoding
        label = self.convert_label(self.labels[ind], self.classes)

        # Return the image and its label
        return (image, label)

    def convert_label(self, label, classes):
        """Convert the numerical label to n-hot encoding.

        Params:
        -------
        - label: a string of conditions corresponding to an image's class

        Returns:
        --------
        - binary_label: (Tensor) a binary encoding of the multi-class label
        """

        binary_label = torch.zeros(len(classes))
        for key, value in classes.items():
            if value in label:
                binary_label[key] = 1.0
        return binary_label
