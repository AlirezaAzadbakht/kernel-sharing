import torch
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import random


class ModifiedCifar10(CIFAR10):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, cutout=0):
        super(ModifiedCifar10, self).__init__(root, train,
                                            transform, target_transform,
                                            download)
        self.cutout = cutout

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super(ModifiedCifar10, self).__getitem__(index)

        if self.train and self.cutout > 1:
            _, h, w = img.size()
            center_x = random.randint(0, w - 1)
            x_min = max(0, center_x - self.cutout // 2)
            x_max = min(w-1, center_x + self.cutout // 2)

            center_y = random.randint(0, h - 1)
            y_min = max(0, center_y - self.cutout // 2)
            y_max = min(h-1, center_y + self.cutout // 2)

            img[:, y_min:y_max, x_min:x_max] *= 0.

        return img, target

class ModifiedCifar100(CIFAR100):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, cutout=0):
        super(ModifiedCifar100, self).__init__(root, train,
                                            transform, target_transform,
                                            download)
        self.cutout = cutout

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super(ModifiedCifar100, self).__getitem__(index)

        if self.train and self.cutout > 1:
            _, h, w = img.size()
            center_x = random.randint(0, w - 1)
            x_min = max(0, center_x - self.cutout // 2)
            x_max = min(w-1, center_x + self.cutout // 2)

            center_y = random.randint(0, h - 1)
            y_min = max(0, center_y - self.cutout // 2)
            y_max = min(h-1, center_y + self.cutout // 2)

            img[:, y_min:y_max, x_min:x_max] *= 0.

        return img, target


class DataLoader(object):
    def __init__(self, aug=True, cutout=0, dataset="cifar10"):
        self.cutout = cutout
        if aug:
            transform_scheme = transforms.Compose([
                transforms.RandomResizedCrop(32,
                                             scale=(0.95, 1.0),
                                             ratio=(0.9, 1.1),
                                             interpolation=2),
                transforms.RandomRotation(10),
                transforms.RandomCrop(32, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_scheme = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])

        test_transform_scheme = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        
        if dataset == "cifar10":
            dataset_class = ModifiedCifar10
        elif dataset == "cifar100":
            dataset_class = ModifiedCifar100
        else:
            raise ValueError()

        # training set
        self.train_set = dataset_class(
            root=f'./data/',
            train=True,
            download=True,
            transform=transform_scheme,
            cutout=cutout)
        self.training_set_size = self.train_set.data.shape[0]

        # test set
        self.test_set = dataset_class(
            root=f'./data/',
            train=False,
            download=True,
            transform=test_transform_scheme,
            cutout=cutout)

    def generator(self, train=True, batch_size=128, GPU_num=4, num_worker=8, CUDA=True):

        def _generator(data_iter):
            for image, label in data_iter:
                if CUDA:
                    image = image.cuda()
                    label = label.cuda()
                yield image, label

        return _generator(torch.utils.data.DataLoader(
            self.train_set if train else self.test_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=train,
            num_workers=num_worker
        ))
