import gzip
import pickle as pkl
import errno
import os
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision.datasets.utils import check_integrity
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
from scipy.io import loadmat

from data_list import ImageList, list_train_test_split


# DATA_PATH = "./data/" # specify your local path to store datasets
DATA_PATH = "/home/data/yzhangdx/dataset"
RANDOM_STATE = 1234


def get_dataset(dataname, data_split='train', use_normalize=False, test_size=0.1, train_size=None):
    '''
    Args:
        dataname: str
        data_split: str, 'train', 'val' or 'test'
        use_normalize: boolean. normalize data to unit gaussian or not. 
        test_size: float, train/val ratio
    '''
    if train_size is None and test_size is not None:
        train_size = 1. - test_size
    elif test_size == 0. and train_size is not None and train_size < 1.:
        test_size = 1 - train_size #dump
    assert(test_size + train_size <= 1.)
    transform_ops = [transforms.ToTensor(), ]
    if dataname == "mnist":
        transform_ops += [transforms.Lambda(lambda x: F.pad(x, (2,2,2,2), 'constant', 0)), 
                          transforms.Lambda(lambda x: x.expand(3,-1,-1))
                          ]
        if use_normalize:
            # transform_ops.append(transforms.Normalize((0.1307,), (0.3081,)))
            # transform_ops.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))) # rescale to [-1,1]
            transform_ops.append(transforms.Normalize((0.5, ), (0.5, ))) # rescale to [-1,1]
        print('transform ops: {}'.format(transform_ops))
        if data_split == "test":
            return datasets.MNIST(os.path.join(DATA_PATH, 'mnist'), train=False, download=True,
                           transform=transforms.Compose(transform_ops))
        else:
            train_ds = datasets.MNIST(os.path.join(DATA_PATH, 'mnist'), train=True, download=True,
                           transform=transforms.Compose(transform_ops))
            if data_split == 'train' and train_size == 1.:
                return train_ds
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(test_size*len(train_ds)), train_size=int(train_size*len(train_ds)), random_state=RANDOM_STATE)
            train_index, val_index = list(skf.split(train_ds.data, train_ds.targets))[0]
            if data_split == 'train':
                return torch.utils.data.Subset(train_ds, train_index)
            else:
                return torch.utils.data.Subset(train_ds, val_index)
    elif dataname == "usps":
        usps = pkl.load(gzip.open(os.path.join(DATA_PATH, 'usps/usps_28x28.pkl'), "rb"), encoding="latin1")
        if data_split in ["train", "val"]:
            # 7438, 1, 28, 28
            images = torch.from_numpy(usps[0][0])
            # 7438x[0~9]
            labels = torch.from_numpy(usps[0][1]).long()
        else: # 1860
            images = torch.from_numpy(usps[1][0])
            labels = torch.from_numpy(usps[1][1]).long()
        images = F.pad(images, (2,2,2,2))
        images = images.expand(-1,3,-1,-1)
        if use_normalize:
            # images = (images - 0.1608) / 0.2578
            # images = (images - 0.1231) / 0.2357
            images = (images - 0.5) / 0.5
        if data_split == "test":
            return torch.utils.data.TensorDataset(images, labels)
        else:
            if data_split == 'train' and train_size == 1.:
                return torch.utils.data.TensorDataset(images, labels)
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(test_size*len(labels)), train_size=int(train_size*len(labels)), random_state=RANDOM_STATE)
            train_index, val_index = list(skf.split(images, labels))[0]
            if data_split == 'train':
                return torch.utils.data.TensorDataset(images[train_index], labels[train_index])
            else:
                return torch.utils.data.TensorDataset(images[val_index], labels[val_index])
    elif dataname == "svhn":
        # always use normalization for svhn, rescale to [-1,1]
        transform_ops.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        if data_split == "test":
            return datasets.SVHN(os.path.join(DATA_PATH, 'svhn'), split=data_split, download=True,
                           transform=transforms.Compose(transform_ops))
        else:
            train_ds = torch.utils.data.ConcatDataset([
                datasets.SVHN(os.path.join(DATA_PATH, 'svhn'), split='train', download=True,
                              transform=transforms.Compose(transform_ops)), 
                datasets.SVHN(os.path.join(DATA_PATH, 'svhn'), split='extra', download=True,
                              transform=transforms.Compose(transform_ops)), 
                ])
            # train_ds = datasets.SVHN(os.path.join(DATA_PATH, 'svhn'), split='train', download=True,
            #                transform=transforms.Compose(transform_ops))
            if data_split == 'train' and train_size == 1.:
                return train_ds
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(test_size*len(train_ds)), train_size=int(train_size*len(train_ds)), random_state=RANDOM_STATE)
            train_index, val_index = list(skf.split(
                np.concatenate([ds.data for ds in train_ds.datasets], 0), 
                np.concatenate([ds.labels for ds in train_ds.datasets], 0)
                ))[0]
            if data_split == 'train':
                return torch.utils.data.Subset(train_ds, train_index)
            else:
                return torch.utils.data.Subset(train_ds, val_index)
    elif dataname == "syn_digits":
        mat_data = loadmat(os.path.join(DATA_PATH, 'SynthDigits', 'synth_{}_32x32.mat'.format(data_split)))
        images = torch.from_numpy(mat_data['X'] / 255.).float()
        labels = torch.from_numpy(mat_data['y']).long().view(-1)
        images = images.permute(3,2,0,1)
        if use_normalize:
            images = (images - 0.5) / 0.5
        if data_split == "test":
            return torch.utils.data.TensorDataset(images, labels)
        else:
            if data_split == 'train' and train_size == 1.:
                return torch.utils.data.TensorDataset(images, labels)
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(test_size*len(labels)), train_size=int(train_size*len(labels)), random_state=RANDOM_STATE)
            train_index, val_index = list(skf.split(images, labels))[0]
            if data_split == 'train':
                return torch.utils.data.TensorDataset(images[train_index], labels[train_index])
            else:
                return torch.utils.data.TensorDataset(images[val_index], labels[val_index])
    elif dataname.startswith("cifar"):
        CIFAR_DATASETS = {"cifar10": datasets.CIFAR10, 
                          "cifar100": datasets.CIFAR100}
        tv_cifar = CIFAR_DATASETS[dataname]
        if data_split == "train":
            transform_ops = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                            ]
        if use_normalize:
            # transform_ops.append(transforms.Normalize((0.4914, 0.4822, 0.4465), 
            #                                           (0.2023, 0.1994, 0.2010)))
            transform_ops.append(transforms.Normalize((0.5, 0.5, 0.5), 
                                                      (0.5, 0.5, 0.5)))
        print('transforms for {} set: {}'.format(data_split, transform_ops))
        # Remap class indices so that the frog class (6) has an index of -1 as it does not appear int the STL dataset
        cls_mapping = np.array([0, 1, 2, 3, 4, 5, -1, 6, 7, 8])
        if data_split == "test":
            test_ds = tv_cifar(os.path.join(DATA_PATH, 'cifar'), train=False, download=True,
                           transform=transforms.Compose(transform_ops))
            if dataname == 'cifar10':
                test_ds.targets = cls_mapping[test_ds.targets]
                valid_indices = test_ds.targets > -1
                test_ds.data, test_ds.targets = test_ds.data[valid_indices], test_ds.targets[valid_indices]
                return test_ds
            else:
                return test_ds
        else:
            train_ds = tv_cifar(os.path.join(DATA_PATH, 'cifar'), train=True, download=True,
                           transform=transforms.Compose(transform_ops))
            if dataname == 'cifar10':
                train_ds.targets = cls_mapping[train_ds.targets]
                valid_indices = train_ds.targets > -1
                train_ds.data, train_ds.targets = train_ds.data[valid_indices], train_ds.targets[valid_indices]
            if data_split == 'train' and train_size == 1.:
                return train_ds
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(test_size*len(train_ds)), train_size=int(train_size*len(train_ds)), random_state=RANDOM_STATE)
            train_index, val_index = list(skf.split(train_ds.data, train_ds.targets))[0]
            # print('training set of cifar: {} samples. '.format(len(train_ds)))
            # print('set sizes: train={}\tval={}'.format(len(train_index), len(val_index)))
            if data_split == 'train':
                return torch.utils.data.Subset(train_ds, train_index)
            else:
                return torch.utils.data.Subset(train_ds, val_index)
    elif dataname == 'stl10':
        '''cifar-10-stl-10 transfer task
           align two label spaces by removing "monkey" class in stl
        '''
        if data_split in ["train", 'val']:
            transform_ops = [transforms.Resize(32),
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                            ]
        else:
            transform_ops = [transforms.Resize(32),
                             transforms.ToTensor(),
                            ]
        if use_normalize:
            transform_ops.append(transforms.Normalize((0.5, 0.5, 0.5), 
                                                      (0.5, 0.5, 0.5)))
        print('transforms for {} set: {}'.format(data_split, transform_ops))
        # remap class indices to match cifar-10
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 6, -1, 7, 8])
        if data_split == "test":
            stl_test_ds = datasets.STL10(os.path.join(DATA_PATH, 'stl10'), split='test', download=True,
                           transform=transforms.Compose(transform_ops))
            stl_test_ds.labels = cls_mapping[stl_test_ds.labels]
            valid_indices = stl_test_ds.labels > -1
            stl_test_ds.data, stl_test_ds.labels = stl_test_ds.data[valid_indices], stl_test_ds.labels[valid_indices]
            return stl_test_ds
        else:
            train_ds = datasets.STL10(os.path.join(DATA_PATH, 'stl10'), split='train', download=True,
                           transform=transforms.Compose(transform_ops))
            train_ds.labels = cls_mapping[train_ds.labels]
            valid_indices = train_ds.labels > -1
            train_ds.data, train_ds.labels = train_ds.data[valid_indices], train_ds.labels[valid_indices]
            if data_split == 'train' and train_size == 1.:
                return train_ds
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(test_size*len(train_ds)), train_size=int(train_size*len(train_ds)), random_state=RANDOM_STATE)
            train_index, val_index = list(skf.split(train_ds.data, train_ds.targets))[0]
            # print('training set of cifar: {} samples. '.format(len(train_ds)))
            # print('set sizes: train={}\tval={}'.format(len(train_index), len(val_index)))
            if data_split == 'train':
                return torch.utils.data.Subset(train_ds, train_index)
            else:
                return torch.utils.data.Subset(train_ds, val_index)
    elif dataname == 'imagenet32x32':
        '''imagenet 32x32 as source domain
        '''
        if data_split in ["train", 'val']:
            transform_ops = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                            ]
        else:
            transform_ops = [transforms.ToTensor(),
                            ]
        if use_normalize:
            transform_ops.append(transforms.Normalize((0.5, 0.5, 0.5), 
                                                      (0.5, 0.5, 0.5)))
        print('transforms for {} set: {}'.format(data_split, transform_ops))
        if data_split == "test":
            return ImageNetDS(os.path.join(DATA_PATH, 'imagenet32x32'), 32, train=False, 
                           transform=transforms.Compose(transform_ops))
        else:
            train_ds = ImageNetDS(os.path.join(DATA_PATH, 'imagenet32x32'), 32, train=True,
                           transform=transforms.Compose(transform_ops))
            if data_split == 'train' and train_size == 1.:
                return train_ds
            skf = StratifiedShuffleSplit(n_splits=1, test_size=int(test_size*len(train_ds)), train_size=int(train_size*len(train_ds)), random_state=RANDOM_STATE)
            train_index, val_index = list(skf.split(train_ds.train_data, train_ds.train_labels))[0]
            # print('training set of cifar: {} samples. '.format(len(train_ds)))
            # print('set sizes: train={}\tval={}'.format(len(train_index), len(val_index)))
            if data_split == 'train':
                return torch.utils.data.Subset(train_ds, train_index)
            else:
                return torch.utils.data.Subset(train_ds, val_index)
    else:
        raise ValueError('no supported loader for dataset {}'.format(dataname))


class ImageNetDS(torch.utils.data.Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.

    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    base_folder = 'Imagenet{}_train'
    train_list = [
        ['train_data_batch_1', ''],
        ['train_data_batch_2', ''],
        ['train_data_batch_3', ''],
        ['train_data_batch_4', ''],
        ['train_data_batch_5', ''],
        ['train_data_batch_6', ''],
        ['train_data_batch_7', ''],
        ['train_data_batch_8', ''],
        ['train_data_batch_9', ''],
        ['train_data_batch_10', '']
    ]

    test_list = [
        ['val_data', ''],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.') # TODO

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pkl.load(fo)
                    self.train_data.append(entry['data'])
                    self.train_labels += [label - 1 for label in entry['labels']]
                    self.mean = entry['mean']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pkl.load(fo)
            self.test_data = entry['data']
            self.test_labels = [label - 1 for label in entry['labels']]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
