"""Script for calculating RA."""

import os
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
from training import dataset
from torchvision import transforms

from predictor import Predictor

#----------------------------------------------------------------------------

def calculate_ra_stats(
    image_path, model_path, expected_class, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):
    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    print('Loading model...')

    # List images.
    print(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, random_seed=seed)
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    predictor = Predictor(mode='clip', device=device)

    # Divide images into batches.
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=max_batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    print(f'Calculating statistics for {len(dataset_obj)} images...')
    classes = []
    for images, _labels in tqdm.tqdm(data_loader, unit='batch'):
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        images = predictor.transform(images)
        logits = predictor.predict(images.to(device))
        pred_classes = torch.argmax(logits, axis=-1)
        classes.append(pred_classes.cpu().detach().numpy())

    classes = np.concatenate(classes)
    assert classes.shape == expected_class.shape, f"{classes.shape}, {expected_class.shape}"

    # print(type(classes), type(expected_class))
    classes = torch.tensor(classes, device=expected_class.device)
    # print(classes, expected_class, classes == expected_class)

    return (classes == expected_class).float().mean()

#----------------------------------------------------------------------------

@click.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--model', 'model_path',      help='Model location', metavar='NPZ|URL',    type=str, required=True, default='cifar_classifier.pth')
@click.option('--class_pkl',  'expected_class',             help='The unlearned (and generated) class', type=str, required=True, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--out_path', 'ra_out_path', help='Path to write RA', metavar='PATH|ZIP',              type=str, default='./results/ra.txt', required=False)

def main(image_path, model_path, expected_class, seed, batch, ra_out_path=None):
    """Calculate RA for a given set of images."""

    with open(expected_class, 'rb') as f:
        expected_class = pickle.load(f)
    print(expected_class.shape)

    ra = calculate_ra_stats(
        image_path=image_path, model_path=model_path, expected_class=expected_class,
        seed=seed, max_batch_size=batch)

    print("RA:", ra)
    if ra_out_path is not None:
        os.makedirs(os.path.dirname(ra_out_path), exist_ok=True)        
        with open(ra_out_path, 'a') as f:
            f.write(f'{ra:g}\n')

#----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------