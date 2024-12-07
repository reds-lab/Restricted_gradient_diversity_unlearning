import argparse
import os

import pandas as pd
import torch
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
import json
from datasets import load_dataset
import numpy as np
from pathlib import Path
import json
import re
from torch.nn.functional import cosine_similarity
import open_clip
import gc
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))
import utils as util

def save_scores(path, scores):
    with open(path + "/alignment_scores.json", "w") as f:
        json.dump(scores, f)


def get_image_paths(folder_path):
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".png")
    ]
    image_files.sort(key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[0]))

    return image_files

def calculate_clip_alignment(
    image_paths,
    prompts,
    clip_model_path="ViT-B-32",
    pretrained_model="laion2b_s34b_b79k",
    batch_size=32,
):
    # Load the CLIP model, preprocess, and tokenizer
    model, preprocess, _ = open_clip.create_model_and_transforms(
        clip_model_path, pretrained=pretrained_model
    )
    tokenizer = open_clip.get_tokenizer(clip_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_scores = []

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_image_paths = image_paths[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]

        # Prepare the images and prompts
        images = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for path in batch_image_paths]
        ).to(device)
        text = tokenizer(batch_prompts).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            # Encode images and text
            image_features = model.encode_image(images)
            text_features = model.encode_text(text)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarities
            cosine_similarities = image_features @ text_features.T

            # Extract scores for corresponding pairs
            scores = cosine_similarities.diag().cpu().tolist()
            all_scores.extend(scores)

    return all_scores

def generate_images(
    model_name,
    proposed,
    proposed_path,
    forget,
    retain,
    save_path,
    device="cuda:0",
    guidance_scale=7.5,
    image_size=512,
    ddim_steps=100,
    num_samples=10,
):
    util.print_section("Generating Images for Evaluation")
    print(util.magenta("Configuration:"))
    print(util.cyan(f"  Model Name: {model_name}"))
    print(util.cyan(f"  Proposed Path: {proposed_path}"))
    print(util.cyan(f"  Forget: {forget}"))
    print(util.cyan(f"  Retain: {retain}"))
    print(util.cyan(f"  Save Path: {save_path}"))

    # Model loading
    print(util.yellow("\nLoading model..."))
    
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.safety_checker = None
                
    if proposed:
        if "SD" not in model_name:
            try:
                model_path = PROJECT_ROOT / "models" / proposed_path / f"{forget}_{retain}" / model_name / f"{model_name.replace('compvis', 'diffusers')}.pt"
                print(util.blue(f"Loading model from: {model_path}"))        
                pipe.unet.load_state_dict(torch.load(model_path))
                print(util.green("Model loaded successfully!"))        
            except Exception as e:
                print(util.red(f"Error loading model: {e}"))
                raise
                
    prompt_file_y_train = "./data/y_train.json"
    prompt_file_y_c_train = "./data/y_c_train.json"
    prompt_file_y_test = "./data/y_test.json"
    prompt_file_y_c_test = "./data/y_c_test.json"

    y_train = []
    with open(prompt_file_y_train, "r") as f:
        y_train = json.load(f)
    y_c_train = []
    with open(prompt_file_y_c_train, "r") as f:
        y_c_train = json.load(f)
    y_test = []
    with open(prompt_file_y_test, "r") as f:
        y_test = json.load(f)
    y_c_test = []
    with open(prompt_file_y_c_test, "r") as f:
        y_c_test = json.load(f)

    if args.proposed:
        folder_path_y_train = f"{save_path}/{proposed_path}/{forget}_{retain}_y_train/"
        os.makedirs(folder_path_y_train, exist_ok=True)
        folder_path_y_c_train = f"{save_path}/{proposed_path}/{forget}_{retain}_y_c_train/"
        os.makedirs(folder_path_y_c_train, exist_ok=True)
        folder_path_y_test = f"{save_path}/{proposed_path}/{forget}_{retain}_y_test/"
        os.makedirs(folder_path_y_test, exist_ok=True)
        folder_path_y_c_test = f"{save_path}/{proposed_path}/{forget}_{retain}_y_c_test/"
        os.makedirs(folder_path_y_c_test, exist_ok=True)
    else:
        folder_path_y_train = f"{save_path}/{model_name}/y_train/"
        os.makedirs(folder_path_y_train, exist_ok=True)
        folder_path_y_c_train = f"{save_path}/{model_name}/y_c_train/"
        os.makedirs(folder_path_y_c_train, exist_ok=True)
        folder_path_y_test = f"{save_path}/{model_name}/y_test/"
        os.makedirs(folder_path_y_test, exist_ok=True)
        folder_path_y_c_test = f"{save_path}/{model_name}/y_c_test/"
        os.makedirs(folder_path_y_c_test, exist_ok=True)

    def generate_and_save_images(prompts, folder, prompt_batch_size=20):
        image_idx = 1
        for i in range(0, len(prompts), prompt_batch_size):
            prompt_batch = prompts[i : i + prompt_batch_size]
            images = pipe(prompt_batch, num_images_per_prompt=num_samples).images
            for image in images:
                image.save(f"{folder}/{image_idx}.png")
                image_idx += 1

    generate_and_save_images(y_train, folder_path_y_train)
    generate_and_save_images(y_c_train, folder_path_y_c_train)
    generate_and_save_images(y_test, folder_path_y_test)
    generate_and_save_images(y_c_test, folder_path_y_c_test)

    # Get paths and prompts for y_train
    print(util.yellow("\nCalculating alignment scores..."))
        
    image_paths_y_train = get_image_paths(folder_path_y_train)
    alignment_scores_y_train = calculate_clip_alignment(image_paths_y_train, y_train)

    # Get paths and prompts for y_c_train
    image_paths_y_c_train = get_image_paths(folder_path_y_c_train)
    alignment_scores_y_c_train = calculate_clip_alignment(image_paths_y_c_train, y_c_train)

    # Get paths and prompts for y_test
    image_paths_y_test = get_image_paths(folder_path_y_test)
    alignment_scores_y_test = calculate_clip_alignment(image_paths_y_test, y_test)

    # Get paths and prompts for y_c_test
    image_paths_y_c_test = get_image_paths(folder_path_y_c_test)
    alignment_scores_y_c_test = calculate_clip_alignment(image_paths_y_c_test, y_c_test)

    mean_y_train = np.mean(np.array(alignment_scores_y_train))
    mean_y_c_train = np.mean(np.array(alignment_scores_y_c_train))
    mean_y_test = np.mean(np.array(alignment_scores_y_test))
    mean_y_c_test = np.mean(np.array(alignment_scores_y_c_test))

    print(util.cyan("\nAlignment Scores:"))
    print(util.cyan(f"  Y Train: {mean_y_train:.4f}"))
    print(util.cyan(f"  Y C Train: {mean_y_c_train:.4f}"))
    print(util.cyan(f"  Y Test: {mean_y_test:.4f}"))
    print(util.cyan(f"  Y C Test: {mean_y_c_test:.4f}"))

    save_scores(folder_path_y_train, alignment_scores_y_train)
    save_scores(folder_path_y_c_train, alignment_scores_y_c_train)
    save_scores(folder_path_y_test, alignment_scores_y_test)
    save_scores(folder_path_y_c_test, alignment_scores_y_c_test)
    print(util.green(f"Saved alignment scores for Y Train to {folder_path_y_train}"))
    print(util.green(f"Saved alignment scores for Y C Train to {folder_path_y_c_train}"))
    print(util.green(f"Saved alignment scores for Y Test to {folder_path_y_test}"))
    print(util.green(f"Saved alignment scores for Y C Test to{folder_path_y_c_test}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages", description="Generate Images using Diffusers Code"
    )
    parser.add_argument("--model_name", help="name of model", type=str, required=True)
    parser.add_argument(
        "--proposed", help="Toggle proposed on or off", action="store_true"
    )
    parser.add_argument("--proposed_path", type=str, default="proposed")
    parser.add_argument(
        "--forget", help="uniform or diverse", type=str, default="uniform"
    )
    parser.add_argument(
        "--retain", help="uniform or diverse", type=str, default="uniform"
    )
    parser.add_argument(
        "--save_path", help="folder where to save images", type=str, required=True
    )
    parser.add_argument(
        "--device",
        help="cuda device to run on",
        type=str,
        required=False,
        default="cuda:0",
    )
    parser.add_argument(
        "--guidance_scale",
        help="guidance to run eval",
        type=float,
        required=False,
        default=7.5,
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--num_samples",
        help="number of samples per prompt",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=100,
    )
    args = parser.parse_args()

    model_name = args.model_name
    proposed = args.proposed
    proposed_path = args.proposed_path
    forget = args.forget
    retain = args.retain
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples = args.num_samples

    generate_images(
        model_name,
        proposed,
        proposed_path,
        forget,
        retain,
        save_path,
        device=device,
        guidance_scale=guidance_scale,
        image_size=image_size,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
    )
