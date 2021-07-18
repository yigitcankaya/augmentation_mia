import numpy as np

import torch
import torch.nn as nn
import random
import os
import sys
import pickle

import matplotlib.pyplot as plt

import models as m

from pathlib import Path
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from opacus import PrivacyEngine, utils

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# for compatibility
class NullScheduler():
    def __init__(self):
        pass
    def step(self):
        pass

# https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, device):
        self.n_holes = n_holes
        self.length = length
        self.device = device


    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        if img.ndim == 4: # batch
            h = img.size(2)
            w = img.size(3)
        else: # single img
            h = img.size(1)
            w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            
        mask = torch.from_numpy(mask).to(self.device)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class ManualData(torch.utils.data.Dataset):
    def __init__(self, data, labels, device='cpu'):
        self.data = torch.from_numpy(data).to(device, dtype=torch.float)
        self.device = device
        self.labels = torch.from_numpy(labels).to(device, dtype=torch.long)
        self.train = True

        self.transforms = None
        self.gaussian_std = None


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.train:
            if self.transforms is not None:
                data = self.transforms(data)
            
            if self.gaussian_std is not None:
                data = torch.clamp(data + torch.randn(data.size(), device=self.device) * self.gaussian_std, min=0, max=1)

        return (data, self.labels[idx])

    def add_crop(self, padding_size):
        if padding_size > 0: # cropping can only be done on the cpu tensors
            self.data = self.data.to('cpu')
            self.labels = self.labels.to('cpu')
            self.device = 'cpu'
            self.transforms = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(self.data.shape[-1], padding=padding_size), transforms.ToTensor()])

    def add_cutout(self, cutout_size):
        if cutout_size > 0:
            self.transforms = transforms.Compose([Cutout(n_holes=1, length=cutout_size, device=self.device)])


    def add_gaussian_aug(self, std_dev):
        self.gaussian_std = std_dev


def get_subset_data(ds, idx=None):
    
    idx = idx if idx is not None else np.arange(len(ds.data))
    np_data = ds.data[idx].cpu().detach().numpy()
    np_labels = ds.labels[idx].cpu().detach().numpy()

    return ManualData(np_data, np_labels, ds.device)


def file_exists(filename):
    return os.path.isfile(filename) 

# for reproducibility
def set_random_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_path(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return path

def parse_model_path(model_path):
    try:
        sections = model_path.split('_')
        params = {}
        params['dset_name'] = sections[0]
        params['laug_type'] = sections[2]
        params['laug_param'] = eval(sections[3])
        params['daug_type'] = sections[5]
        params['daug_param'] = eval(sections[6])
        params['dp_norm_clip'] = eval(sections[9])
        params['dp_noise'] = eval(sections[11])
        params['num_epochs'] = int(sections[13])
        params['path_suffix'] = sections[15]
        return params
    except:
        return None

def get_random_seeds():
    return [0,1,2] # 0, 1 and 2


def get_reduction_params():
    names = ['median', 'min', 'max', 'std', 'mean']
    funcs = [np.median, np.min, np.max, np.std, np.mean]
    return names, funcs 


def get_ds_and_clf(ds_name, is_dp=False, num_epochs=35, return_dataset=True, device='cpu'):
    
    
    if num_epochs == 3:
        milestones = [1,2]

    elif num_epochs == 4:
        milestones = [2,3]

    elif num_epochs == 7:
        milestones = [3, 6]

    elif num_epochs == 35:
        milestones = [20, 30]
    
    else:
        milestones = [int(num_epochs/2), int(2*num_epochs/3)]


    if ds_name == 'fmnist':
        clf = m.FMNISTClassifier(num_classes=10, dp=is_dp, device=device)
        if return_dataset:
            datasets = get_fmnist_datasets(device=device)

    elif ds_name == 'cifar10':
        clf = m.CIFARClassifier(num_classes=10, dp=is_dp, device=device)
        if return_dataset:
            datasets = get_cifar10_datasets(device=device)

    elif ds_name == 'cifar100':
        clf = m.CIFARClassifier(num_classes=100, dp=is_dp, device=device)
        if return_dataset:
            datasets = get_cifar100_datasets(device=device)
    
    if return_dataset:
        return clf, datasets, milestones
    else:
        return clf, milestones


def get_ds(ds_name, device='cpu'):
    if ds_name == 'fmnist':
        datasets = get_fmnist_datasets(device=device)

    elif ds_name == 'cifar10':
        datasets = get_cifar10_datasets(device=device)

    elif ds_name == 'cifar100':
        datasets = get_cifar100_datasets(device=device)

    return datasets

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n  
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            acc = float(correct_k.mul_(100.0 / batch_size))
            res.append(acc)
        return res

def get_pytorch_device():
    device = 'cpu'
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = 'cuda'
    return device


def get_loaders(ds, shuffle=True, batch_size=128, device='cpu'):

    if device == 'cpu':
        num_workers = 4
    else:
        num_workers = 0

    train_ds, test_ds = ds

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,  num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_loader(dataset, shuffle=True, batch_size=128, device='cpu'):

    if device == 'cpu':
        num_workers = 4
    else:
        num_workers = 0


    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
    return loader


def get_cifar10_datasets(device='cpu'):
    create_path('data/cifar10')
    t = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=t)
    test_dataset = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=t)
        
    train_data, test_data = (train_dataset.data/ 255) , (test_dataset.data / 255)
    train_data, test_data = train_data.transpose((0, 3, 1, 2)), test_data.transpose((0,3,1,2))  
    train_labels, test_labels = np.array(train_dataset.targets), np.array(test_dataset.targets)
    
    train_dataset = ManualData(train_data, train_labels, device)
    test_dataset = ManualData(test_data, test_labels, device)
    
    
    return train_dataset, test_dataset

def get_cifar100_datasets(device='cpu'):
    create_path('data/cifar100')
    t = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=t)
    test_dataset = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=t)
        
    train_data, test_data = (train_dataset.data/ 255) , (test_dataset.data / 255)
    train_data, test_data = train_data.transpose((0, 3, 1, 2)), test_data.transpose((0,3,1,2))  
    train_labels, test_labels = np.array(train_dataset.targets), np.array(test_dataset.targets)
    
    train_dataset = ManualData(train_data, train_labels, device)
    test_dataset = ManualData(test_data, test_labels, device)
    
    return train_dataset, test_dataset

def get_fmnist_datasets(device='cpu'):
    create_path('data/fmnist')
    t = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST('data/fmnist', train=True, download=True, transform=t)
    test_dataset = datasets.FashionMNIST('data/fmnist', train=False, download=True, transform=t)

    train_data, test_data = (train_dataset.data.numpy() / 255) , (test_dataset.data.numpy() / 255)
    train_labels, test_labels = np.array(train_dataset.targets), np.array(test_dataset.targets)
    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])

    train_dataset = ManualData(train_data, train_labels, device)
    test_dataset = ManualData(test_data, test_labels, device)

    return train_dataset, test_dataset

def get_std_optimizer(model, milestones=None, wd=1e-6, optim_type='adam', lr=None):

    if optim_type == 'adam':
        lr = 0.001 if lr is None else lr
        print('Adam Optimizer')
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.999), amsgrad=True, weight_decay=wd)
    
    elif optim_type == 'sgd':
        lr = 0.1 if lr is None else lr        
        print('SGD Optimizer')
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1, momentum=0.9, weight_decay=wd)

    if milestones is None:
        print('Null Scheduler')
        scheduler = NullScheduler()
    else:
        print(f'Scheduler w/ milestone: {milestones}')
        scheduler = MultiStepLR(optimizer, milestones=milestones)

    return optimizer, scheduler


def modify_optim_for_dp(model, optimizer, norm_clip=1.0, noise_mult=0.01, batch_size=64, accumulation_steps=4, training_size=50000):
    privacy_engine = PrivacyEngine(model,
        batch_size=batch_size * accumulation_steps,
        sample_size=training_size,
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=noise_mult,
        max_grad_norm=norm_clip
    )

    privacy_engine.attach(optimizer)

    return privacy_engine

def load_model(model_path, device='cpu'):
    model = torch.load(model_path+'.dat', map_location=device)
    return model

def save_model(model_path, model):
    with open(model_path+'.dat', 'wb') as f:
        torch.save(model, f)


def loader_inst_counter(loader):
    num_instances = 0
    for batch in loader:
        num_instances += len(batch[1])
    return num_instances  


def loader_batch_counter(loader):
    num_batches = 0
    for _ in loader:
        num_batches += 1
    return num_batches      
    
def collect_all_models(models_path):
    model_params = []

    for dir in [os.path.join(models_path, d) for d in os.listdir(models_path)]:
        params = parse_model_path(os.path.basename(dir))

        if params is None:
            continue

        mpath = os.path.join(dir, 'clf')

        if not file_exists(mpath + '.dat'):
            continue
        
        params['model_path'] = mpath
        params['dir'] = dir
        model_params.append(params)

        if params['dp_norm_clip'] != 0 and params['dp_noise'] != 0:
            params['dp'] = True
            params['epsilon'] = load_model(mpath).dp_epsilons[-1]

        else:
            params['dp'] = False
            params['epsilon'] = np.inf


    return model_params




def sample_batches(loader, frac_batches):
    num_batches = loader_batch_counter(loader)

    if frac_batches < 1.0:
        sampled_batches = random.sample(list(range(num_batches-1)), int(frac_batches*num_batches))
        num_instances = len(sampled_batches)*loader.batch_size
    if isinstance(frac_batches, int):
        sampled_batches = random.sample(list(range(num_batches-1)), frac_batches)
        num_instances = len(sampled_batches)*loader.batch_size
    else:
        sampled_batches = list(range(num_batches))
        num_instances = af.loader_inst_counter(loader)
    
    return sampled_batches, num_instances

    
def save_batch_of_tensor_images(save_path, data, nrow_mult=1):
    save_image(data, nrow=16*nrow_mult, fp=f'{save_path}.png', normalize=True, range=(0,1), padding=4, pad_value=255)
