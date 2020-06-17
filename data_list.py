#from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
from sklearn.model_selection import train_test_split


def make_dataset(image_list, labels):
    if labels is not None:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i][0]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def list_train_test_split(image_list, train_size):
    '''split the dataset into labeled and unlabeled sets
    Args: 
        image_list: [list], each element is a str, "image_file_path label"
        num_labels_per_class: [int], number of labeled sample per class, if -1, then return a labeled dataset
    Returns:
        labeled_images: list of str, path to labeled images
        labeled_y: np.array with shape (num_samples,1)
        unlabeled_images
        unlabeled_y
    '''
    if train_size == 1.:
        return [val.split()[0] for val in image_list], np.array([int(val.split()[1]) for val in image_list]).reshape(-1,1), None, None
    elif train_size == 0.:
        return None, None, [val.split()[0] for val in image_list], np.array([int(val.split()[1]) for val in image_list]).reshape(-1,1)
    else:
        images = [val.split()[0] for val in image_list]
        labels = [int(val.split()[1]) for val in image_list]
        assert(len(images) == len(labels))
        # print('image size={}, label size={}'.format(len(images),len(labels)))
        num_classes = len(np.unique(labels))
        train_images, test_images, train_y, test_y = train_test_split(images, labels, 
            train_size=train_size, stratify=labels, random_state=1)
        return train_images, np.array(train_y).reshape(-1,1), test_images, np.array(test_y).reshape(-1,1)


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
