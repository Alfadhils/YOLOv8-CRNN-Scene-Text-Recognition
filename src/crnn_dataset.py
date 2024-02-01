import os
import numpy as np 
import csv

import torch
from PIL import Image

torch.manual_seed(42)

class TRDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch dataset for text recognition. Converts images and labels into appropriate format for training.
    """

    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    
    def __init__(self, root_dir=None, labels=None, images=None, paths=None, img_height=32, img_width=100):
        """
        Initialize the dataset. Use root_dir and labels for train/val, use paths for inference.

        Parameters:
            root_dir (str): Root directory containing image and label data.
            labels (str): File name of the CSV file containing image labels.
            paths (list): List of image paths (optional, if provided, labels parameter is ignored).
            img_height (int): Height of the input images.
            img_width (int): Width of the input images.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.img_paths = None
        self.labels = None
        self.image = False

        if paths:
            self.img_paths = paths
        elif images :
            self.img_paths = images
            self.image = True
        else:
            self.img_paths, self.labels = self.read_labels(root_dir, labels)

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Get a specific sample from the dataset. Used for dataloader.

        Parameters:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the image tensor and, if labels are available, target and target length tensors.
        """
        if self.image:
            image = self.img_paths[idx]
        else :
            img_path = self.img_paths[idx]
        
            try:
                image = Image.open(img_path)
            except IOError:
                print('Corrupted image for %d' % idx)
                return self[idx + 1]
        
        image = preprocess(image, self.img_height, self.img_width)
        
        if self.labels:
            label = self.labels[idx]
            target = [self.CHAR2LABEL[c] for c in label]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image

    def read_labels(self, root_dir=None, labels=None):
        """
        Read image labels from a CSV file.

        Parameters:
            root_dir (str): Root directory containing image and label data.
            labels (str): File name of the CSV file containing image labels.

        Returns:
            tuple: A tuple containing lists of image paths and corresponding labels.
        """
        img_data, label_data = [], []
        with open(os.path.join(root_dir, labels), "r") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                img_name = row[0]
                img_path = os.path.join(root_dir, 'images', img_name + '.jpg')
                label = row[1].lower()
                img_data.append(img_path)
                label_data.append(label)
        return img_data, label_data
    
def collate_batch(batch):
    """
    Collate a batch of samples into a single batch. Required for 

    Parameters:
        batch (list): List of samples, where each sample is a tuple containing image, target, and target length.

    Returns:
        tuple: A tuple containing the collated image tensor, target tensor, and target length tensor.
    """
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths

def preprocess(img, height, width):
    """
    Preprocess image
    
    Parameters:
    - img (PIL.Image): The input image
    - height (int): Desired height of the preprocessed image
    - width (int): Desired width of the preprocessed image
    
    Returns:
    - torch.FloatTensor: Preprocessed image tensor
    """
    
    image = img.convert('L')
    image = image.resize((width, height), resample=Image.BILINEAR)
    image = np.array(image)
    image = image.reshape((1, height, width))
    image = (image / 127.5) - 1.0
    return torch.FloatTensor(image)

def get_split(root_dir=None, labels=None, set=None, img_width=100, img_height=32, batch_size=64, splits=[0.98, 0.01, 0.01]):
    """
    Create data loaders for training, validation, and testing.

    Parameters:
        root_dir (str): Root directory containing image and label data.
        labels (str): File name of the CSV file containing image labels.
        img_width (int): Width of the input images.
        img_height (int): Height of the input images.
        batch_size (int): Batch size for the data loaders.
        splits (list): List containing the proportion of data for training, validation, and testing.

    Returns:
        tuple: A tuple containing data loaders for training, validation, and testing.
    """
    dataset = TRDataset(root_dir=root_dir,
                        labels=labels,
                        img_width=img_width,
                        img_height=img_height)
    
    train_size, val_size, test_size = int(splits[0] * len(dataset)), int(splits[1] * len(dataset)), int(splits[2] * len(dataset))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    
    if set == 'train' :
        return train_loader
    elif set == 'val' :
        return val_loader
    elif set == 'test' :
        return test_loader
    
    return train_loader, val_loader, test_loader
