
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
cp -r Restricted_gradient_diversity_unlearning/SD/ldm/* ldm/
cp -r Restricted_gradient_diversity_unlearning/SD/configs/* configs/

# Add directories that do not exist originally (train-scripts, eval-scripts, data)
cp -r Restricted_gradient_diversity_unlearning/SD/train-scripts ./
cp -r Restricted_gradient_diversity_unlearning/SD/eval-scripts ./
cp -r Restricted_gradient_diversity_unlearning/SD/data ./

# Add additional files to the main stable-diffusion directory
cp Restricted_gradient_diversity_unlearning/SD/diffusers_unet_config.json ./
cp Restricted_gradient_diversity_unlearning/SD/run-surgery.sh ./
cp Restricted_gradient_diversity_unlearning/SD/utils.py ./
```

### 4. Run 

```bash
  ./run-surgery.sh
```


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




