# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from torch_utils import pcgrad

#----------------------------------------------------------------------------
from torch.utils.data import DataLoader, SubsetRandomSampler
import math
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
#----------------------------------------------------------------------------

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

def apply_pcgrad_update(unlearn_grads, retain_grads      ):
    """
    Apply PCGrad logic to resolve conflicts and update the model gradients.
    After performing the gradient surgery, scale the updated retain gradients by lambda_.
    """
    flat_unlearn_grad, unlearn_shapes = flatten_grads(unlearn_grads)
    flat_retain_grad, _ = flatten_grads(retain_grads)

    # Apply projection
    updated_unlearn_grad = project_grad_onto(flat_unlearn_grad, flat_retain_grad)
    updated_retain_grad = project_grad_onto(flat_retain_grad, flat_unlearn_grad)

    # Now, unflatten the updated gradients back to their original shapes
    unlearn_grads_updated = unflatten_grads(updated_unlearn_grad, unlearn_shapes)
    retain_grads_updated = unflatten_grads(updated_retain_grad, unlearn_shapes)

    return unlearn_grads_updated, retain_grads_updated

def select_balanced_samples(dataset, num_samples_per_class=10):
    indices_per_class = {i: [] for i in range(10)}  # Assuming 10 classes for CIFAR-10
    for idx, (_, label) in enumerate(dataset):
        # Convert label to numpy array if it's a PyTorch tensor
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        class_id = np.argmax(label)  # Identify class ID from one-hot encoding
        indices_per_class[class_id].append(idx)

    selected_indices = []
    for class_id, class_indices in indices_per_class.items():
        if len(class_indices) >= num_samples_per_class:  # Ensure there are enough samples
            selected_indices.extend(np.random.choice(class_indices, num_samples_per_class, replace=False))
        else:  # In case some class has fewer than `num_samples_per_class` samples
            selected_indices.extend(np.random.choice(class_indices, len(class_indices), replace=False))
            print(f"Warning: Class {class_id} has less than {num_samples_per_class} samples.")

    return selected_indices

def count_samples_per_class(dataset, num_classes=10):
    class_counts = {i: 0 for i in range(num_classes)}
    for _, label in dataset:
        # Assuming label is a PyTorch tensor or a NumPy array
        if isinstance(label, torch.Tensor):
            label = label.numpy()  # Convert to numpy array if it's a tensor
        class_id = np.argmax(label)  # Find the index (class ID) with the '1'
        class_counts[class_id] += 1
    return class_counts

def count_samples_per_class_subset(dataset, selected_indices, num_classes=10):
    # Initialize count dictionary
    class_counts = {i: 0 for i in range(num_classes)}
    # Iterate over selected indices to count classes
    for idx in selected_indices:
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.numpy()  # Convert to numpy if it's a tensor
        class_id = np.argmax(label)
        class_counts[class_id] += 1
    return class_counts

def verify_loader_balance_general(loader, expected_samples_per_class=50, total_classes=10):
    # Initialize class counts for all classes
    class_counts = {i: 0 for i in range(total_classes)}

    for _, labels in loader:
        # Assuming labels are one-hot encoded; adjust if different
        labels = labels.argmax(dim=1).cpu().numpy()
        for label in labels:
            class_counts[label] += 1

    # Check if all classes have the expected number of samples, except for one class which should have 0
    correct_sample_distribution = sum(count == expected_samples_per_class for count in class_counts.values()) == total_classes - 1 and sum(count == 0 for count in class_counts.values()) == 1

    print(f"All classes, except one, have exactly {expected_samples_per_class} samples, and one class has 0 samples:", correct_sample_distribution)

    # For diagnostic purposes, print out counts
    zero_sample_classes = [class_id for class_id, count in class_counts.items() if count == 0]
    print("Class counts:", class_counts)
    if zero_sample_classes:
        print(f"Class with 0 samples (likely the target class): {zero_sample_classes[0]}")
    else:
        print("No class with 0 samples found. There might be an issue with the dataset distribution.")
        
def get_class_samples_rules(classes_to_forget):
    sample_from = {
        0: [(2, 'bird', 225), (8, 'ship', 225)],
        2: [(4, 'deer', 225), (6, 'frog', 225)],
        5: [(3, 'cat', 225), (6, 'frog', 225)]
    }
    # Get rules for specified forget classes
    rules = []
    for class_id in classes_to_forget:
        rules.extend(sample_from.get(class_id, []))
    return rules
    
def select_custom_samples(dataset, classes_to_forget):
    sample_rules = get_class_samples_rules(classes_to_forget)
    indices_per_class = {}
    for rule in sample_rules:
        class_id, _, num_samples = rule
        indices_per_class.setdefault(class_id, (num_samples, []))

    # Collect indices for relevant classes
    for idx, (_, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        class_id = np.argmax(label)
        if class_id in indices_per_class:
            indices_per_class[class_id][1].append(idx)

    selected_indices = []
    for class_id, (num_samples, indices) in indices_per_class.items():
        if len(indices) >= num_samples:
            selected_indices.extend(np.random.choice(indices, num_samples, replace=False))
        else:
            selected_indices.extend(np.random.choice(indices, len(indices), replace=False))
            print(f"Warning: Not enough samples in class {class_id}. Needed {num_samples}, got {len(indices)}.")

    return selected_indices

def print_class_sample_counts(loader):
    class_counts = {}
    for _, labels in loader:
        # Assuming labels are one-hot encoded; adjust if different
        labels = labels.argmax(dim=1).cpu().numpy()
        for label in labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

    # Print the counts for each class
    total_classes = 10  # Adjust as necessary based on your dataset
    for class_id in range(total_classes):
        count = class_counts.get(class_id, 0)
        print(f"Class {class_id}: {count} samples")

# +
def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    unlearn_alg         = 'RGD',     # [UNLEARN] Algorithm for unlearning
    unlearn_lambda      = 1.,       # [UNLEARN] Lambda term trading off between forget and retain losses
    unlearn_alpha       = 1e-2,      # [UNLEARN] Truncation value
    classes             = [],        # [UNLEARN] Classes to forget
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    print("dataset_kwargs", dataset_kwargs)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    if 'assym_batch' in unlearn_alg:
        unlearn_alg = '_'.join(unlearn_alg.split('_')[:-2])
        forget_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=int(batch_gpu // 9), **data_loader_kwargs))
    else:
        forget_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    if unlearn_alg != 'normal':
        retain_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs, retain=True) # subclass of training.dataset.Dataset
        retain_sampler = misc.InfiniteSampler(dataset=retain_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
        retain_iterator = iter(torch.utils.data.DataLoader(dataset=retain_obj, sampler=retain_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    #--------------------------------------------------------------------------------#

    # Assuming retain_obj is your dataset object
    class_counts = count_samples_per_class(retain_obj)
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} samples")

    #--------------------------------------------------------------------------------#

    if unlearn_alg == 'GradDiff' or unlearn_alg == 'RG' or unlearn_alg == 'RGD':
        selected_indices = select_balanced_samples(retain_obj, num_samples_per_class=50)
        # selected_indices = select_custom_samples(retain_obj, classes)
        retain_sampler = SubsetRandomSampler(selected_indices)
        retain_loader = DataLoader(
            dataset=retain_obj,
            sampler=retain_sampler,
            batch_size=batch_gpu,  # Ensure this is suitably set
            **data_loader_kwargs  # Ensure these kwargs are appropriate for DataLoader
        )

        # Assuming retain_obj is your dataset object and selected_indices is defined
        class_counts = count_samples_per_class_subset(retain_obj, selected_indices)
        for class_id, count in class_counts.items():
            print(f"Class {class_id}: {count} samples")

        if unlearn_alg == 'GradDiff' or unlearn_alg == 'RG':
            retain_loader_cy = cycle(retain_loader)

        # Assuming retain_loader is defined and constructed
        print_class_sample_counts(retain_loader)    
        verify_loader_balance_general(retain_loader, expected_samples_per_class=50, total_classes=10)
    
    #--------------------------------------------------------------------------------#

    num_iterations = 0

    #--------------------------------------------------------------------------------#

    while True:
        
        #--------------------------------------------------------------------------------#
        num_iterations += 1
        #--------------------------------------------------------------------------------#

        assert num_accumulation_rounds == 1, 'Currently we only support num_accumulation_rounds == 1!'
        forget_images, forget_labels = next(forget_iterator)
        def acc_grad(images, labels, lambda_scaling=1, zero_grad=True, truncate=False):
            if zero_grad:
                optimizer.zero_grad(set_to_none=True)
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)
            loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe, truncate=truncate, t_alpha=unlearn_alpha)
            training_stats.report('Loss/loss', loss)
            loss.sum().mul(lambda_scaling * loss_scaling / batch_gpu_total).backward()
        
            for param in net.parameters():
                norms = []
                if param.grad is not None:
                    norms.append(param.grad.norm())
                    torch.nan_to_num(param.grad, nan=0, posinf=1e3, neginf=-1e3, out=param.grad)
        
            return loss.mean().item()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr']
            # g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)


        if unlearn_alg == 'GradDiff':
            loss = acc_grad(forget_images, forget_labels, truncate=True)
            ascent_grads = [-param.grad.detach() for param in net.parameters()]
            optimizer.zero_grad()

            retain_images, retain_labels = next(retain_loader_cy) ## if you want to use subset (450 samples)
            # retain_images, retain_labels = next(retain_iterator) ## if you want to use 45K images
            retain_loss = acc_grad(retain_images, retain_labels)
            descent_grads = [param.grad.detach() for param in net.parameters()]

            for param, ascent_grad, descent_grad in zip(net.parameters(), ascent_grads, descent_grads):
                if param.grad is not None:
                    combined_grad = (ascent_grad) + (unlearn_lambda * descent_grad)
                    param.grad.data = combined_grad.data

            optimizer.step()
            loss = (loss, retain_loss)

        elif unlearn_alg == 'RG':
            loss = acc_grad(forget_images, forget_labels, truncate=True)
            ascent_grads_before_surgery = [-param.grad.detach() for param in net.parameters()]
            optimizer.zero_grad()

            retain_images, retain_labels = next(retain_loader_cy) ## if you want to use subset (450 samples)
            # retain_images, retain_labels = next(retain_iterator) ## if you want to use 45K images
            retain_loss = acc_grad(retain_images, retain_labels)
            descent_grads = [param.grad.detach() for param in net.parameters()]

            updated_ascent_grads, updated_descent_grads = apply_pcgrad_update(ascent_grads_before_surgery, descent_grads)

            for param, ascent_grad, descent_grad in zip(net.parameters(), updated_ascent_grads, updated_descent_grads):
                if param.grad is not None:
                    combined_grad = (ascent_grad) + (unlearn_lambda * descent_grad)
                    param.grad.data = combined_grad.data

            optimizer.step()
            loss = (loss, retain_loss)

        elif unlearn_alg == 'RGD':
            loss = acc_grad(forget_images, forget_labels, truncate=True)
            ascent_grads_before_surgery = [-param.grad.detach().detach() for param in net.parameters()]
            optimizer.zero_grad()

            descent_grads_accum = [torch.zeros_like(param) for param in net.parameters()]
            retain_loss = 0.
            num_batches = 0  # Keep track of how many batches are processed

            # Iterate over the retain batches
            for r_images, r_labels in retain_loader: # 450 samples
                optimizer.zero_grad()
                r_loss = acc_grad(r_images, r_labels)
                retain_loss += r_loss
                # Accumulate gradients from retain batches
                for accum_grad, param in zip(descent_grads_accum, net.parameters()):
                    accum_grad += param.grad.detach()
                num_batches += 1

            for grad in descent_grads_accum:
                grad /= num_batches

            updated_ascent_grads, updated_descent_grads = apply_pcgrad_update(ascent_grads_before_surgery, descent_grads_accum)

            for param, updated_ascent_grad, updated_descent_grad in zip(net.parameters(), updated_ascent_grads, updated_descent_grads):
                if param.grad is not None:
                    updated_grad = updated_ascent_grad + (unlearn_lambda * updated_descent_grad)
                    param.grad.data = updated_grad.data

            optimizer.step()
            loss = (loss, retain_loss/num_batches)

        else:
            raise NotImplementedError

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        # assert np.allclose(ema_beta, 1.), f"ema_beta: {ema_beta}"
        ema_beta = 1.
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            # p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            p_ema.copy_(p_net.detach())
            assert torch.allclose(p_ema, p_net), f"{(p_ema - p_net).norm().item()}"

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        if isinstance(loss, tuple):
            fields += ['losses'] + [f"{l:<2.4f}" for l in loss]
        else:
            fields += [f"loss {loss:<2.4f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            if unlearn_alg == 'compute_saliency_mask':
                process_gradients(gradients, classes)

            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

def process_gradients(gradients, classes):
    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        assert len(classes) == 1
        mask_path = os.path.join('./training-runs/salun_masks', str(classes[0]))
        os.makedirs(mask_path, exist_ok=True)

        threshold_list = [0.5]
        for i in threshold_list:
            print(i)
            sorted_dict_positions = {}
            hard_dict = {}

            # Concatenate all tensors into a single tensor
            all_elements = - torch.cat(
                [tensor.flatten() for tensor in gradients.values()]
            )

            # Calculate the threshold index for the top 10% elements
            threshold_index = int(len(all_elements) * i)

            # Calculate positions of all elements
            positions = torch.argsort(all_elements)
            ranks = torch.argsort(positions)

            start_index = 0
            for key, tensor in gradients.items():
                num_elements = tensor.numel()
                tensor_ranks = ranks[start_index : start_index + num_elements]

                sorted_positions = tensor_ranks.reshape(tensor.shape)
                sorted_dict_positions[key] = sorted_positions

                # Set the corresponding elements to 1
                threshold_tensor = torch.zeros_like(tensor_ranks)
                threshold_tensor[tensor_ranks < threshold_index] = 1
                threshold_tensor = threshold_tensor.reshape(tensor.shape)
                hard_dict[key] = threshold_tensor
                start_index += num_elements

            torch.save(hard_dict, os.path.join(mask_path, f'with_{str(i)}.pt'))
