# Adaptive Batch Normalization (AdaBN) in PyTorch

This repository contains an implementation of **Adaptive Batch Normalization (AdaBN)** in PyTorch, a technique that adapts batch normalization statistics to better generalize across domain shifts.

[Revisiting Batch Normalization For Practical Domain Adaptation](https://arxiv.org/abs/1603.04779)

## Files in the Repository

- **`batchnorm_adapt.py`**  
  Main script that demonstrates AdaBN usage. 

- **`utils.py`**  
  Contains the core AdaBN implementation including the hook class and statistics computation functions.

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch 1.10+
- NumPy

Install the required packages:
```bash
pip install torch numpy
```

### Usage

```python
from utils import compute_bn_stats, replace_bn_stats

# Set model to eval mode
model.eval()

# Compute target domain statistics
bn_stats = compute_bn_stats(model, target_dataloader)

# Apply AdaBN
replace_bn_stats(model, bn_stats)
```

### Running the Example

```bash
python batchnorm_adapt.py
```

## How AdaBN Works

AdaBN adapts models to new domains by:
- Computing BatchNorm statistics on target domain data
- Replacing source domain statistics with target domain statistics
- Keeping all learned weights unchanged

This requires no additional training or parameters.