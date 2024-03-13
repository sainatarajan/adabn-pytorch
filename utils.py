import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet18
from torchvision.models.vgg import vgg16_bn, vgg11_bn
from functools import partial


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Just trying to seed everything so I don't find myself looking confused at the screen
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Fake dataset class. Trying to be as fake as it can be
class ImageGeneratorDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.vector_dim = (3, 128, 128)
        self.data = []
        self.create_data()

    def create_data(self):
        for i in range(self.num_samples):
            self.data.append(torch.zeros(self.vector_dim))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


# Simple model to understand the behavior of AdaBN passing through two BN layers
class BasicModel(nn.Module):

    def __init__(self, ):
        super(BasicModel, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm2d(num_features=64)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)

        return x


# The hook class is responsible to store the BN outputs when the test dataloader is passed
class BatchNormStatHook(object):
    """
    Hook to accumulate statistics from BatchNorm layers during inference.
    """

    def __init__(self):
        self.bn_stats = {}  # Dictionary to store layer name and accumulated statistics

    def __call__(self, module, input, output, name):
        """
        Hook function called during the forward pass of BatchNorm layers.

        Args:
            module (nn.Module): The BatchNorm layer.
            input (torch.Tensor): Input tensor to the layer.
            output (torch.Tensor): Output tensor from the layer.
        """
        layer_name = name
        # Check if layer name already exists (multiple BN layers with same type)
        # But I think this might not be required if the model is well defined properly?
        # Not taking care of nn.Sequential

        if layer_name not in self.bn_stats:
            self.bn_stats[layer_name] = {'mean': 0, 'var': 0, 'count': 0}

        # Ensure output is not a view (avoid potential errors)
        output = output.clone()  # Create a copy of the output

        # Calculate mean and variance of the output
        mean = output.mean([0, 2, 3])
        var = output.var([0, 2, 3], unbiased=False)

        # Update accumulated statistics for this layer
        self.bn_stats[layer_name]['mean'] += mean.sum()
        self.bn_stats[layer_name]['var'] += var.sum()

        # This might not be required, but still saving just in-case
        self.bn_stats[layer_name]['count'] += mean.numel()


def compute_bn_stats(model, dataloader):
    """
    Computes mean and variance of BatchNorm layer outputs across all images in the dataloader.

    Args:
      model (nn.Module): The trained model.
      dataloader (torch.utils.data.DataLoader): The dataloader for the data.

    Returns:
      dict: Dictionary containing layer names and their mean and variance statistics.
    """

    # Create a hook instance
    hook = BatchNormStatHook()

    # Register the hook on all BatchNorm layers in the model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.register_forward_hook(partial(hook, name=name))

    # Iterate through the dataloader
    with torch.no_grad():
        for data in dataloader:
            # Forward pass (hook will accumulate statistics)
            model(data.to(device))

    # Calculate mean and variance for each layer
    for layer_name, stats in hook.bn_stats.items():
        # print("Found the layer!!!")
        mean = stats['mean'] / stats['count']
        var = stats['var'] / stats['count']
        hook.bn_stats[layer_name] = {'mean': mean, 'var': var}

    # Return the accumulated statistics
    return hook.bn_stats


# Now replace the current stats with the computed one
def replace_bn_stats(model, bn_stats):
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in bn_stats and isinstance(module, nn.BatchNorm2d):
                print('Before---------------------------------------')
                print(module.running_mean)
                module.running_mean.data.copy_(bn_stats[name]['mean'])
                module.running_var.data.copy_(bn_stats[name]['var'])
                print(module.running_mean)
                print('After---------------------------------------')