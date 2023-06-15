import glob
import glob
import os
from os import rmdir
from shutil import move

import torch
from torchvision import datasets, transforms


# Based on https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/val_format.py
def TINYIMAGENET(root, train=True, transform=None, target_transform=None, download=True):
    def _exists(root, filename):
        return os.path.exists(os.path.join(root, filename))

    def _download(url, root, filename):
        datasets.utils.download_and_extract_archive(url=url,
                                                    download_root=root,
                                                    extract_root=root,
                                                    filename=filename)

    def _setup(root, base_folder):
        target_folder = os.path.join(root, base_folder, 'val/')

        val_dict = {}
        with open(target_folder + 'val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]

        paths = glob.glob(target_folder + 'images/*')
        paths[0].split('/')[-1]
        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(target_folder + str(folder)):
                os.mkdir(target_folder + str(folder))

        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            dest = target_folder + str(folder) + '/' + str(file)
            move(path, dest)

        os.remove(target_folder + 'val_annotations.txt')
        rmdir(target_folder + 'images')

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    base_folder = 'tiny-imagenet-200'

    if download and not _exists(root, filename):
        _download(url, root, filename)
        _setup(root, base_folder)
    folder = os.path.join(root, base_folder, 'train' if train else 'val')

    return datasets.ImageFolder(folder, transform=transform, target_transform=target_transform)


''' Dataset '''


def dimension(dataset):
    if dataset == 'mnist':
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == 'cifar10':
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == 'cifar100':
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == 'tiny_imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes


def _get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    transform.append(transforms.Resize(size=size))
    return transforms.Compose(transform)


def dataloader(root, dataset, batch_size, train, workers=8, length=None, imsize=None):
    # Dataset
    if imsize is None:
        size_dict = {
            'mnist': 28,
            'cifar10': 32,
            'cifar100': 32,
            'tiny_imagenet': 64,
            'imagenet': 224
        }
        imsize = size_dict.get(dataset, None)
    kwargs = {}
    if dataset == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        transform = _get_transform(size=imsize, padding=0, mean=mean, std=std, preprocess=False)
        dataset = datasets.MNIST(root, train=train, download=True, transform=transform)
    elif dataset == 'cifar10':
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
        transform = _get_transform(size=imsize, padding=4, mean=mean, std=std, preprocess=train and imsize != 224)
        root = f'{root}/CIFAR10'
        dataset = datasets.CIFAR10(root, train=train, download=True, transform=transform)
    elif dataset == 'cifar100':
        mean, std = (0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
        root = f'{root}/CIFAR100'
        transform = _get_transform(size=imsize, padding=4, mean=mean, std=std, preprocess=train and imsize != 224)
        dataset = datasets.CIFAR100(root, train=train, download=True, transform=transform)
    elif dataset == 'tiny_imagenet':
        mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
        transform = _get_transform(size=imsize, padding=4, mean=mean, std=std, preprocess=train and imsize != 224)
        dataset = TINYIMAGENET(root, train=train, download=True, transform=transform)
    elif dataset == 'imagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(imsize, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(imsize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        folder = '{}/{}'.format(root, 'train' if train else 'val')
        dataset = datasets.ImageFolder(folder, transform=transform)

    # Dataloader
    use_cuda = torch.cuda.is_available()
    kwargs.update({'num_workers': workers, 'pin_memory': False} if use_cuda else {})
    shuffle = train is True
    if length is not None:
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             **kwargs)

    return dataloader
