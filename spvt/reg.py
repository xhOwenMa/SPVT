import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_LiRPA.bound_ops import *


def compute_lp_reg(model, latent_ptb_tensor, X_batch, target_norm=1.0):
    """
    Compute a gradient penalty that encourages the model to be locally Lipschitz.
    
    Args:
        model: the BoundedModule that takes (latent, state) as inputs and outputs the steering angle.
        latent_ptb_tensor: BoundedTensor for the latent variable.
        X_batch: state tensor of shape [batch_size, ...]. Must be differentiable.
        target_norm: target gradient norm (default 1.0).
    Returns:
        penalty
    """
    X_batch.requires_grad_(True)
    latent_ptb_tensor.requires_grad_(True)
    output = model(latent_ptb_tensor, X_batch)
    grad_outputs = torch.ones_like(output)
    gradients_state = torch.autograd.grad(
        outputs=output,
        inputs=X_batch,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients_latent = torch.autograd.grad(
        outputs=output,
        inputs=latent_ptb_tensor,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients_state = gradients_state.view(gradients_state.size(0), -1)
    gradients_latent = gradients_latent.view(gradients_latent.size(0), -1)
    
    grad_norm_state = torch.norm(gradients_state, dim=1)
    grad_norm_latent = torch.norm(gradients_latent, dim=1)
    penalty_state = (grad_norm_state - target_norm).pow(2).mean()
    penalty_latent = (grad_norm_latent - target_norm).pow(2).mean()

    return penalty_state, penalty_latent


def compute_reg(model, tol=0.5):
    """ Computes regularization losses for ReLU stability activation balance. """
    loss = torch.zeros(()).to(next(model.parameters()).device)
    loss_std, loss_relu = torch.zeros_like(loss), torch.zeros_like(loss)
    cnt = 0
    for name, module in model._modules.items():
        # if isinstance(module, BoundRelu) and name in names:
        if isinstance(module, BoundRelu):
            lower, upper = module.inputs[0].lower, module.inputs[0].upper
            center = (upper + lower) / 2
            diff = ((upper - lower) / 2)
            mean_ = center.mean()
            std_ = center.std()
            
            loss_std += F.relu(tol - std_) / tol
            
            mask_act, mask_inact = lower > 0, upper < 0
            mean_act = (center * mask_act).mean()
            mean_inact = (center * mask_inact).mean()
            delta = (center - mean_)**2
            var_act = (delta * mask_act).sum()
            var_inact = (delta * mask_inact).sum()
            
            mean_ratio = mean_act / -mean_inact
            var_ratio = var_act / var_inact
            mean_ratio = torch.min(mean_ratio, 1 / mean_ratio.clamp(min=1e-12))
            var_ratio = torch.min(var_ratio, 1 / var_ratio.clamp(min=1e-12))
            loss_relu_ = ((F.relu(tol - mean_ratio) + F.relu(tol - var_ratio)) / tol)
            
            if not torch.isnan(loss_relu_) and not torch.isinf(loss_relu_):
                loss_relu += loss_relu_
            
            cnt += 1
    
    if cnt > 0:
        loss_std /= cnt
        loss_relu /= cnt
    
    return loss_std + loss_relu