## Requirements
Tested with CUDA 12.8, Ubuntu 24.04

## Installation

#### Create environment
```
conda create -n exoskeleton -y python=3.13
conda activate exoskeleton
```

#### Install PyTorch
https://pytorch.org/get-started/locally/

#### Install other dependencies
`pip install -r requirements.txt`

# idk if we'll use this because pytorch has to be installed per-device to have cuda enabled, and this doesnt account for that
#### Updating Dependencies
Install pipreqs `pip install pipreqs`
After installing new dependencies with pip, do `pipreqs ./` in the root directory of the repo.
From there, you can do `pip install -r requirements.txt`

