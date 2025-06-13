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
            # FIXED: Create some variation instead of all zeros
            self.data.append(torch.randn(self.vector_dim) * 0.1)

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
            # FIXED: Initialize with proper structure for accumulation
            self.bn_stats[layer_name] = {'mean': 0, 'var': 0, 'count': 0}

        # CRITICAL FIX #1: Process INPUT tensor, not output!
        # Ensure output is not a view (avoid potential errors)
        x = input[0].clone()  # FIXED: Use input[0] instead of output

        # Calculate mean and variance of the INPUT (not output)
        mean = x.mean([0, 2, 3])  # FIXED: Process input activations
        var = x.var([0, 2, 3], unbiased=False)

        # CRITICAL FIX #2: Do NOT sum across channels! Keep per-channel info
        batch_size = x.size(0)

        # Initialize accumulators with correct shape on first call
        if isinstance(self.bn_stats[layer_name]['mean'], int):
            self.bn_stats[layer_name]['mean'] = torch.zeros_like(mean)
            self.bn_stats[layer_name]['var'] = torch.zeros_like(var)

        # Update accumulated statistics for this layer (keep per-channel)
        self.bn_stats[layer_name]['mean'] += mean * batch_size  # FIXED: No .sum()!
        self.bn_stats[layer_name]['var'] += var * batch_size  # FIXED: No .sum()!

        # This might not be required, but still saving just in-case
        self.bn_stats[layer_name]['count'] += batch_size  # FIXED: Count samples, not channels


def compute_bn_stats(model, dataloader):
    """
    Computes mean and variance of BatchNorm layer outputs across all images in the dataloader.

    Args:
      model (nn.Module): The trained model.
      dataloader (torch.utils.data.DataLoader): The dataloader for the data.

    Returns:
      dict: Dictionary containing layer names and their mean and variance statistics.
    """

    # CRITICAL FIX #3: Set model to eval mode, not train mode!
    original_mode = model.training
    model.eval()  # FIXED: Changed from train() to eval()

    # Create a hook instance
    hook = BatchNormStatHook()
    hook_handles = []

    # Register the hook on all BatchNorm layers in the model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            handle = module.register_forward_hook(partial(hook, name=name))
            hook_handles.append(handle)  # FIXED: Store handles for cleanup

    try:
        # Iterate through the dataloader
        with torch.no_grad():
            for data in dataloader:
                # Forward pass (hook will accumulate statistics)
                model(data.to(device))

        # Calculate mean and variance for each layer
        final_stats = {}
        for layer_name, stats in hook.bn_stats.items():
            # print("Found the layer!!!")
            if stats['count'] > 0:
                mean = stats['mean'] / stats['count']  # FIXED: Now divides tensors properly
                var = stats['var'] / stats['count']
                final_stats[layer_name] = {'mean': mean, 'var': var}

    finally:
        # FIXED: Clean up hooks to prevent memory leaks
        for handle in hook_handles:
            handle.remove()
        model.train(original_mode)  # Restore original mode

    # Return the accumulated statistics
    return final_stats


# Now replace the current stats with the computed one
def replace_bn_stats(model, bn_stats):
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in bn_stats and isinstance(module, nn.BatchNorm2d):
                # FIXED: Add shape verification
                expected_shape = module.running_mean.shape
                computed_mean = bn_stats[name]['mean']
                computed_var = bn_stats[name]['var']

                if computed_mean.shape != expected_shape:
                    raise ValueError(f"Shape mismatch for {name}: expected {expected_shape}, got {computed_mean.shape}")

                print('Before---------------------------------------')
                print(module.running_mean)
                module.running_mean.data.copy_(computed_mean.to(module.running_mean.device))  # FIXED: Handle device
                module.running_var.data.copy_(computed_var.to(module.running_var.device))
                print(module.running_mean)
                print('After---------------------------------------')