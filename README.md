# Adaptive Batch Normalization (AdaBN) in PyTorch

This repository contains an implementation of **Adaptive Batch Normalization (AdaBN)** in PyTorch, a technique that adapts batch normalization statistics to better generalize across domain shifts and other dataset-specific variations.

[Revisiting Batch Normalization For Practical Domain Adaptation](https://arxiv.org/abs/1603.04779)

---

## Files in the Repository

- **`batchnorm_adapt.py`**  
  Implements the core functionality of Adaptive Batch Normalization (AdaBN). This includes adapting batch normalization statistics to new datasets or domains. 

- **`utils.py`**  
  Contains utility functions supporting the main implementation, such as data preprocessing, logging, and model manipulation.

---

## Key Features

- **Adaptation to Domain Shifts**  
  AdaBN updates batch normalization statistics to match the target domain, improving model generalization in scenarios like domain adaptation and test-time adaptation.

- **Easy Integration**  
  Modular implementation to easily integrate into existing PyTorch projects.

- **Reusable Utilities**  
  Helper functions for common operations related to adaptive normalization and data handling.

---

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch 1.10+
- NumPy

Install the required packages:
```bash
pip install torch numpy
