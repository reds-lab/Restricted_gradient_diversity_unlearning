import numpy as np
import torch

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

def apply_pcgrad_update(model, unlearn_grads, retain_grads):
    """
    Apply PCGrad logic to resolve conflicts and update the model gradients.
    """
    # First, flatten and combine all grads for comparison
    flat_unlearn_grad, unlearn_shapes = flatten_grads(unlearn_grads)
    flat_retain_grad, _ = flatten_grads(retain_grads)

    # Apply projection
    updated_unlearn_grad = project_grad_onto(flat_unlearn_grad, flat_retain_grad)
    updated_retain_grad = project_grad_onto(flat_retain_grad, flat_unlearn_grad)

    # Now, unflatten the updated gradients back to their original shapes
    unlearn_grads_updated = unflatten_grads(updated_unlearn_grad, unlearn_shapes)
    retain_grads_updated = unflatten_grads(updated_retain_grad, unlearn_shapes)

    # Finally, set the updated gradients back to the model parameters
    for grad, param in zip(unlearn_grads_updated + retain_grads_updated, model.parameters()):
        param.grad = grad