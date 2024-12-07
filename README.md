# Boosting Alignment for Post-Unlearning Text-to-Image Generative Models

**NeurIPS 2024**

[![arXiv](https://img.shields.io/badge/arXiv-2301.10120-red.svg)](--)
[![Venue: NeurIPS 2024](https://img.shields.io/badge/Venue-NeurIPS%202024-blue.svg)](https://nips.cc/)


This is the official implementation for our NeurIPS 2024 paper  
**"Boosting Alignment for Post-Unlearning Text-to-Image Generative Models"**.

The code structure of this project is adapted from the [Saliency Unlearning](https://github.com/OPTML-Group/Unlearn-Saliency/tree/master/SD) codebase.

## Requirements

**Conda Environment and PyTorch:**
1. Create and activate a Conda environment:
    ```bash
    conda create -n unlearning-diffusion python=3.11.5
    conda activate unlearning-diffusion
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

### 1. Base Repository Setup
1. Clone the original Stable Diffusion repository:
    ```bash
    git clone https://github.com/CompVis/stable-diffusion.git
    cd stable-diffusion
    ```

### 2. Model Weights
Download the required model weights (e.g., Stable Diffusion v1.4) and place them into .models/ldm/. 
Please refer to the [Saliency Unlearning repository](https://github.com/OPTML-Group/Unlearn-Saliency/tree/master/SD) for detailed instructions on obtaining the model weights.

### 3. Our Implementation
Add our custom implementation on top of the original stable-diffusion codebase:

```bash
# Clone our repository
git clone https://github.com/reds-lab/Restricted_gradient_diversity_unlearning.git

# Replace the 'ldm' and 'configs' directories with our versions
cp -r Restricted_gradient_diversity_unlearning/ldm/* stable-diffusion/ldm/
cp -r Restricted_gradient_diversity_unlearning/configs/* stable-diffusion/configs/

# Add directories that do not exist originally (train-scripts, eval-scripts, data)
cp -r Restricted_gradient_diversity_unlearning/train-scripts stable-diffusion/
cp -r Restricted_gradient_diversity_unlearning/eval-scripts stable-diffusion/
cp -r Restricted_gradient_diversity_unlearning/data stable-diffusion/

# Add additional files to the main stable-diffusion directory
cp Restricted_gradient_diversity_unlearning/diffusers_unet_config.json stable-diffusion/
cp Restricted_gradient_diversity_unlearning/run-surgery.sh stable-diffusion/
cp Restricted_gradient_diversity_unlearning/utils.py stable-diffusion/
```

### 4. Run 

```bash
  ./run-surgery.sh
```

## Contact

For any questions, issues, or inquiries, please contact [myeongseob@vt.edu](mailto:myeongseob@vt.edu).


## Troubleshooting

**Error:**
```bash
ImportError: cannot import name 'VectorQuantizer2' from 'taming.modules.vqvae.quantize' (/home/myeongseob/miniconda3/envs/unlearning-diffusion/lib/python3.11/site-packages/taming/modules/vqvae/quantize.py)

Replace your local quantize.py with the version from:
https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
```

```bash
ModuleNotFoundError: No module named 'pytorch_lightning.utilities.distributed'

./ldm/models/diffusion/ddpm.py:

Change : from pytorch_lightning.utilities.distributed import rank_zero_only
To : from pytorch_lightning.utilities.rank_zero import rank_zero_only
```

RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
```bash

./ldm/models/diffusion/ddpm.py 

Change  :  logvar_t = self.logvar[t].to(self.device)
To :    t = t.to('cpu')
        logvar_t = self.logvar[t].to(self.device)
```

## Citation

If you find this repository or the ideas presented in our paper useful, please consider citing:

```bibtex
@article{heng2024boosting,
  title={Boosting Alignment for Post-Unlearning Text-to-Image Generative Models},
  author={Heng, Alvin and Soh, Harold},
  journal={arXiv preprint arXiv:2301.10120},
  year={2024}
}

