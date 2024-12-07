import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
from convertModels import savemodelDiffusers, savemodelDiffusers_rgd
from dataset import (
    setup_forget_nsfw_data_rgd,
    setup_forget_nsfw_data,
    setup_model,
)
from diffusers import LMSDiscreteScheduler
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm
import gc
import json
import copy
import utils as util

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def flatten_grads(grads):
    """
    Flatten gradients for all parameters and pack them into a single tensor.
    Also, return the original shapes to enable unflattening later.
    """
    flat_grads = []
    shapes = []
    for grad in grads:
        # Assuming `grads` are already gradient tensors, no need to access `.grad`
        flat_grad = grad.view(-1)
        flat_grads.append(flat_grad)
        shapes.append(grad.size())
    return torch.cat(flat_grads), shapes


def unflatten_grads(grads, shapes):
    """
    Unflatten a single gradient tensor into the original shapes.
    """
    unflattened_grads = []
    i = 0
    for shape in shapes:
        length = np.prod(shape)
        unflattened_grads.append(grads[i:i+length].view(shape))
        i += length
    return unflattened_grads


def project_grad_onto(grad, onto):
    """
    Adjusted to consider all parameters simultaneously for more accurate projection.
    """
    similarity = torch.dot(grad.flatten(), onto.flatten()) / (torch.norm(onto.flatten()) ** 2)
    if similarity.item() < 0:
        return grad - similarity * onto
    return grad


def apply_pcgrad_update(unlearn_grads, retain_grads):
    """
    Apply PCGrad logic to resolve conflicts and update the model gradients.
    After performing the gradient surgery, scale the updated retain gradients by lambda_.
    """
    flat_unlearn_grad, unlearn_shapes = flatten_grads(unlearn_grads)
    flat_retain_grad, _ = flatten_grads(retain_grads)
    
    # Apply projection
    updated_unlearn_grad = project_grad_onto(flat_unlearn_grad, flat_retain_grad) #i
    updated_retain_grad = project_grad_onto(flat_retain_grad, flat_unlearn_grad) #j 
    
    # Now, unflatten the updated gradients back to their original shapes
    unlearn_grads_updated = unflatten_grads(updated_unlearn_grad, unlearn_shapes)
    retain_grads_updated = unflatten_grads(updated_retain_grad, unlearn_shapes)
        
    return unlearn_grads_updated, retain_grads_updated

def nsfw_proposed(
    train_method,
    truncate,
    surgery,
    proposed_path,
    forget,
    retain,    
    c_guidance,
    alpha,
    m,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
):
    util.print_section("Training Stable Diffusion Unlearning")
    print(util.magenta(f"Configuration:"))
    print(util.cyan(f"  - Proposed Path: {proposed_path}"))
    print(util.cyan(f"  - Forget: {forget}"))
    print(util.cyan(f"  - Retain: {retain}"))
    print(util.cyan(f"  - Alpha: {alpha}"))
    print(util.cyan(f"  - M: {m}"))
    
    # During training
    print(util.yellow("\nStarting training..."))
    
    # After saving
    print(util.green(f"Model saved successfully to {folder_path}"))
    
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    forget_dl, remain_dl = setup_forget_nsfw_data_rgd(batch_size, image_size, forget, retain)

    # set model to train
    model.train()
    losses = []    
    retain_losses = []  
    
    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            # print(name)
            parameters.append(param)

    optimizer = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    name = f"compvis-nsfw-method_{train_method}-lr_{lr}"

    # TRAINING CODE
    for epoch in range(epochs):
        with tqdm(total=len(forget_dl)) as time:
            for i, ((forget_images, forget_prompts), (x_retain, y_retain)) in enumerate(zip(forget_dl, remain_dl)):

                optimizer.zero_grad()
                
                # ---------------------------------------------------------------------------------------------------- #  
                ### forget term 
                
                null_prompts = [""] * len(forget_images)

                forget_batch = {"jpg": forget_images.permute(0, 2, 3, 1), "txt": list(forget_prompts)}
                forget_loss = model.shared_step(forget_batch)[0]

                # ---------------------------------------------------------------------------------------------------- #  
                
                if truncate:
                    if forget_loss.item() > m :
                        forget_loss = (forget_loss / torch.norm(forget_loss))  ## to make the loss = 1
                    else:
                        forget_loss = forget_loss
                else:
                    forget_loss = forget_loss

                # ---------------------------------------------------------------------------------------------------- # surgery 

                forget_loss.backward()  # Backward pass to compute gradients
                ascent_grads_before_surgery = [-param.grad.detach() for _, param in model.model.diffusion_model.named_parameters()]
                optimizer.zero_grad()                

                # ---------------------------------------------------------------------------------------------------- #
                ### retain term                 
                 
                retain_batch = {"jpg": x_retain.permute(0, 2, 3, 1), "txt": list(y_retain)}
                retain_loss = model.shared_step(retain_batch)[0]
            
                retain_loss.backward()                
                retain_grads_before_surgery = [param.grad.detach() for _, param in model.model.diffusion_model.named_parameters()]

                if surgery:
                    updated_ascent_grads, updated_descent_grads = apply_pcgrad_update(ascent_grads_before_surgery, retain_grads_before_surgery)
                
                    for (_, param), ascent_grad, descent_grad in zip(model.model.diffusion_model.named_parameters(), updated_ascent_grads, updated_descent_grads):
                        if param.grad is not None:
                            combined_grad = (ascent_grad) + alpha*(descent_grad) 
                            param.grad.data = combined_grad.data.clone()
    
                else:
                    for (_, param), ascent_grad, descent_grad in zip(model.model.diffusion_model.named_parameters(), ascent_grads_before_surgery, retain_grads_before_surgery):
                        if param.grad is not None:
                            combined_grad = (ascent_grad) + alpha*(descent_grad) 
                            param.grad.data = combined_grad.data.clone()                    
                    
                loss = forget_loss + alpha*retain_loss
                
                # ---------------------------------------------------------------------------------------------------- #     
                
                losses.append(forget_loss.item())
                retain_losses.append(retain_loss.item())
                
                optimizer.step()

                time.set_description("Epoch %i" % epoch)
                time.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time.update(1)

    model.eval()
    save_model(
        model,
        name,
        proposed_path,
        forget,
        retain, 
        None,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )

    save_history(losses, retain_losses, name, proposed_path, forget, retain)

def save_model(
    model,
    name,
    proposed_path,
    forget,
    retain,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=True,
    save_diffusers=True,
):
    # SAVE MODEL
    folder_path = PROJECT_ROOT / "models" / proposed_path / f"{forget}_{retain}" / name
    folder_path.mkdir(parents=True, exist_ok=True)
    
    if num is not None:
        path = folder_path / f"{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers_revised(
            name, proposed_path, forget, retain, compvis_config_file, diffusers_config_file, device=device
        )

def save_history(forget_losses, retain_losses, name, proposed_path, forget, retain):
    # SAVE MODEL
    folder_path = PROJECT_ROOT / "models" / proposed_path / f"{forget}_{retain}" / name
    folder_path.mkdir(parents=True, exist_ok=True)
    
    with open(folder_path / "forget_loss.txt", "w") as f:
        f.writelines([str(i) for i in forget_losses])   
    plot_loss(forget_losses, folder_path / "forget_loss.png", n=3)

    with open(folder_path / "retain_loss.txt", "w") as f:
        f.writelines([str(i) for i in retain_losses])     
    plot_loss(retain_losses, folder_path / "retain_loss.png", n=3)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TrainESD",
        description="Finetuning stable diffusion model to erase concepts using ESD method",
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--truncate",
        help="Enable truncation",
        action="store_true",  # This sets the argument to True when it is specified
        default=False         # Default value is False if the argument is not specified
    ) 
    parser.add_argument(
        "--surgery",
        help="Enable truncation",
        action="store_true",  # This sets the argument to True when it is specified
        default=False         # Default value is False if the argument is not specified
    )     
    parser.add_argument(
        "--proposed_path",
        type=str,
        default="proposed_path",
    )        
    parser.add_argument(
        "--forget",
        help="uniform or diverse",
        type=str,
        required=True,
        default="uniform",
    )              
    parser.add_argument(
        "--retain",
        help="uniform or diverse",
        type=str,
        required=True,
        default="uniform",
    )        
    parser.add_argument(
        "--c_guidance",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=7.5,
    )    
    parser.add_argument(
        "--alpha",
        help="weight on descent direction",
        type=float,
        required=False,
        default=1.5,
    )
    parser.add_argument(
        "--m",
        help="truncation value",
        type=float,
        required=False,
        default=1.5,
    )    
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=1
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=int,
        required=False,
        default=1e-5
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="./configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="./models/ldm/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="0",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    args = parser.parse_args()
    
    train_method = args.train_method
    truncate = args.truncate
    surgery = args.surgery    
    proposed_path = args.proposed_path
    forget = args.forget        
    retain = args.retain    
    c_guidance = args.c_guidance
    alpha = args.alpha
    m = args.m
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    nsfw_proposed(
        train_method=train_method,
        truncate=truncate,
        surgery=surgery,
        proposed_path=proposed_path,
        forget=forget,
        retain=retain,                
        c_guidance=c_guidance,
        alpha=alpha,
        m=m,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        config_path=config_path,
        ckpt_path=ckpt_path,
        diffusers_config_path=diffusers_config_path,
        device=device,
        image_size=image_size,
        ddim_steps=ddim_steps,
    )  

