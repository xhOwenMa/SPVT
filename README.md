# Learning Vision-Based Neural Network Controllers with Semi-Probabilistic Safety Guarantees

This repository contains the implementation of our paper "Learning Vision-Based Neural Network Controllers with Semi-Probabilistic Safety Guarantees." Our approach integrates formal neural network verification tools into training for learning a controller with inherent safety guarantees.

## DEMO Video

Check out our demo video for IROS 2025 submission:

[![Watch the video](https://img.youtube.com/vi/ojoJi8951SU/1.jpg)](https://youtu.be/ojoJi8951SU)

## TODO Lists

- [ ] integrate `xplane` experiments into the training scripts

## Contributions

- **Semi-Probabilistic Safety Framework**: A novel approach that bridges the gap between deterministic formal methods and statistical analysis, requiring only verification on a sampled set of datapoints in the state space to have meaningful confidence bounds for k-step safety guarantees over the entire state space
- **Disjoint Training Sets**: A training procedure that maintains disjoint and dynamic sets of ordinary training data and hard examples to maintain and improve control performances and safety at the same time
- **Verification (Safety)-Aware Training**: Integration of formal verification bounds directly into gradient-based training

## Semi-Probabilistic Verification

folder `spv/` contains the verification results to reproduce the figures in our paper. 

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU

### Setup
Clone the repository:
```bash
git clone https://github.com/xhOwenMa/SPVT.git
cd SPVT
```

We recommend creating a virtual environment with conda using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate spvt
```


## Repository Structure

```
├── data/                   # Dataset storage
├── model/                  # Model definitions
│   └── pretrained_ckpts/   # Pretrained checkpoints
└── spvt/                   # Core implementation
    ├── logs/               # Training logs and tensorboard files
    ├── args.py             # Command line arguments
    ├── logger.py           # Logging utilities
    ├── loss.py             # Loss functions
    ├── reg.py              # Regularization methods
    ├── scripts.py          # Support functions for training
    └── utils.py            # Utility functions
├── train.py            # Main training loop
```

## Usage

To train the controller with default parameters:

```bash
python train.py
```

`args.py` contains all the hyperparameters and training options that can be customized.

<!-- ## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{author2023learning,
  title={Learning Vision-Based Neural Network Controllers with Semi-Probabilistic Safety Guarantees},
  author={Author, A. and Author, B.},
  journal={Conference/Journal Name},
  year={2023}
}
``` -->
