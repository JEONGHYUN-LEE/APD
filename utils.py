import torch
import torchvision
from torchvision.transforms import transforms
import models
import random
import numpy as np


def get_models(model_name):
    if model_name == 'resnet18':
        return models.ResNet18()


def get_pruned_models(model_name, factor):
    if model_name == 'resnet18':
        return models.resnet18(factor)


def get_dataloader(dataset_name, batch_size=512, seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    trainloader, testloader = None, None
    if dataset_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
                transforms.ToTensor()
        ])

        def _init_fn(worker_id):
            np.random.seed(seed)

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, pin_memory=True,
                                                  num_workers=50, worker_init_fn=_init_fn)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, pin_memory=True,
                                                 num_workers=50, worker_init_fn=_init_fn)


    return trainloader, testloader
