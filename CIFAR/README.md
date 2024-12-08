
The code structure of this project is adapted from the [Elucidating the Design Space of Diffusion-Based Generative Models](https://github.com/NVlabs/edm) codebase.

## Requirements

**Conda Environment and PyTorch:**
1. Create and activate a Conda environment:
    ```bash
    conda create -n unlearning-cifar python=3.11.5
    conda activate unlearning-cifar
    ```

2. Install PyTorch and CUDA toolkit:
    ```bash
    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

**Python Packages:**
- After setting up the Conda environment and PyTorch, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Installation

### 1. Downloading the necessary files.
Download the required model weights with 
```bash
    wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

Download the data with 

and place them into ./models and ./data respectively.
    
### 2. Our Implementation

```bash
# Clone our repository
git clone https://github.com/reds-lab/Restricted_gradient_diversity_unlearning.git

```

## Run 

```bash
  ./scripts/run-surgery.sh
```

