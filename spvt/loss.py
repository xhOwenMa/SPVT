import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from utils import dynamics, convert_action
from reg import compute_lp_reg, compute_reg

def compute_loss(args, model, x, z, k, buffer):
    """
    Calculate the composite loss for training the safety verification model.
    
    This function computes multiple loss components:
    1. Prediction loss: How well the model matches the expected control actions
    2. Safety loss: Related to the cross-track error bounds
    3. Bound loss: The gap between lower and upper bounds from verification
    4. Regularization losses: To improve model properties
    
    Args:
        args: Training arguments
        model: The bounded neural network model
        x: State tensor (batch_size, state_dim) with cross-track and heading errors
        z: Perturbed latent input tensor (batch_size, latent_dim)
        k: Number of steps to simulate for verification and safety loss
        buffer: Buffer for storing hard examples
        
    Returns:
        tuple containing:
            - total_loss: Combined loss for backpropagation
            - pred_acc_loss: Prediction accuracy loss
            - safety_loss: Safety constraint violation loss
            - bound_gap_loss: Verification bound gap loss
            - reg_loss: ReLU stability regularization loss
            - state_lp_loss: State-space Lipschitz regularization loss
            - latent_lp_loss: Latent-space Lipschitz regularization loss
            - buffer: Updated hard example buffer
            - added_indices: Indices of examples added to the buffer
    """
    assert k >= args.pred_loss_step, f'k should be at least args.pred_loss_step, but got k={k} and args.pred_loss_step={args.pred_loss_step}'

    # Simulate trajectory with bounded verification
    violation_mask, cte_mins, cte_maxs, states_traj, model_controls = sim_trajectory(args, model, z, x, threshold=80.0, K=k)
    
    # 1. Prediction loss - measures control accuracy
    y = calculate_action(states_traj[:, 0:args.pred_loss_step, :])
    pred_acc_loss = nn.MSELoss(reduction='mean')(model_controls[:, 0:args.pred_loss_step, :], y)
    
    # 2. Safety loss - based on cross-track error bounds
    safety_loss_items = (torch.abs(cte_mins[-1]) + torch.abs(cte_maxs[-1])) / (k - 1)
    safety_loss = torch.mean(safety_loss_items)/100  # convert centimeters to meters
    
    # 3. Bound loss - measures verification precision
    bound_gap = calc_bound_loss(args, model, x, z, y)
    bound_gap_loss = nn.L1Loss()(bound_gap, torch.zeros_like(bound_gap))
    
    # 4. Regularization losses - improve model properties
    reg_loss = compute_reg(model)  # ReLU stability regularization
    lp_penalty_state, lp_penalty_latent = compute_lp_reg(model, z, x)  # Lipschitz regularization
    
    # 5. Update hard example buffer with difficult examples and track which were added
    added_indices = []
    if args.buffer_criteria == 'safety_loss':
        added_indices = buffer.add_examples(x, safety_loss_items.detach(), violation_mask)
    elif args.buffer_criteria == 'bound_gap':
        added_indices = buffer.add_examples(x, bound_gap.detach(), violation_mask)
    
    # 6. Compute weighted total loss
    if args.buffer_criteria == 'safety_loss':
        total_loss = (1 - args.dynamic_lambda) * args.lambda_safety * safety_loss
    elif args.buffer_criteria == 'bound_gap':
        total_loss = (1 - args.dynamic_lambda) * args.lambda_bound * bound_gap_loss
    
    # Add other loss components with their respective weights
    total_loss += (args.dynamic_lambda * args.lambda_accuracy * pred_acc_loss + 
                  args.lambda_reg * reg_loss + 
                  args.lambda_lp * (lp_penalty_state + lp_penalty_latent))
    
    return total_loss, pred_acc_loss, safety_loss, bound_gap_loss, reg_loss, lp_penalty_state, lp_penalty_latent, buffer, added_indices

def calc_bound_loss(args, model, x, z, y):
    """
    Calculate bound loss for a batch

    Args:
        args: arguments
        model: model
        x: state (batch_size, state_dim)
        z: purturbed latent input (batch_size, latent_dim)
        y: target action (batch_size, action_dim)
    """
    lb, ub = model.compute_bounds(x=(z, x), method=args.bound_method)
    bound_gap = ub - lb
    return bound_gap

def sim_trajectory(args, model, z_ptb_tensor, x, threshold=80, K=5):
    """
    Args:
        model: BoundedModule
        z_ptb_tensor: BoundedTensor for latent variable
        x: states with shape [batch_size, 2]
        threshold: safety threshold (centimeters)
        K: number of steps to simulate
        loss_type: 'cumulative' or 'barrier' or 'single_step'
    """
    batch_size = x.shape[0]
    cte = x[:, 0]
    he = x[:, 1]
    cte_mins, cte_maxs = [], []
    true_states, true_controls = [], []
    state_cte_min = cte.clone()
    state_cte_max = cte.clone()
    state_he_min = he.clone()
    state_he_max = he.clone()
    violation_mask = torch.zeros((batch_size, K), dtype=torch.bool, device=x.device)

    for step in range(K):
        true_state = torch.stack([cte, he], dim=1) 
        true_states.append(true_state)
        state_bounds_L = torch.stack([state_cte_min, state_he_min], dim=1)
        state_bounds_U = torch.stack([state_cte_max, state_he_max], dim=1)
        state_perturbation = PerturbationLpNorm(
            norm=np.inf,
            x_L=state_bounds_L,
            x_U=state_bounds_U
        )
        state_ptb_tensor = BoundedTensor(true_state, state_perturbation)

        true_control = model(z_ptb_tensor, state_ptb_tensor)
        true_controls.append(true_control)
        lb, ub = model.compute_bounds(x=(z_ptb_tensor, state_ptb_tensor), method=args.bound_method)
        # unnormalize actions (from [-1. 1] to respective range in degrees)
        true_control = convert_action(true_control, 'unnormalize')
        lb = convert_action(lb, 'unnormalize')
        ub = convert_action(ub, 'unnormalize')

        # next state regions
        cte_next_lb_min, he_next_lb_min = dynamics(state_cte_min, state_he_min, lb)
        cte_next_lb_max, he_next_lb_max = dynamics(state_cte_max, state_he_max, lb)
        cte_next_ub_min, he_next_ub_min = dynamics(state_cte_min, state_he_min, ub)
        cte_next_ub_max, he_next_ub_max = dynamics(state_cte_max, state_he_max, ub)
        
        # next state region bounds
        next_cte_min = torch.minimum(
            torch.minimum(cte_next_lb_min, cte_next_lb_max),
            torch.minimum(cte_next_ub_min, cte_next_ub_max)
        )
        next_cte_max = torch.maximum(
            torch.maximum(cte_next_lb_min, cte_next_lb_max),
            torch.maximum(cte_next_ub_min, cte_next_ub_max)
        )
        next_he_min = torch.minimum(
            torch.minimum(he_next_lb_min, he_next_lb_max),
            torch.minimum(he_next_ub_min, he_next_ub_max)
        )
        next_he_max = torch.maximum(
            torch.maximum(he_next_lb_min, he_next_lb_max),
            torch.maximum(he_next_ub_min, he_next_ub_max)
        )

        state_cte_min, state_cte_max = next_cte_min, next_cte_max
        state_he_min, state_he_max = next_he_min, next_he_max
        cte_mins.append(state_cte_min)
        cte_maxs.append(state_cte_max)

        margin_low = threshold - torch.abs(cte_maxs[step])
        margin_high = threshold - torch.abs(cte_mins[step])
        violation_mask[:, step] = (margin_low <= 0) | (margin_high <= 0)

        # next true state
        cte, he = dynamics(cte, he, true_control)
    
    true_states_tensor = torch.stack(true_states)
    true_controls_tensor = torch.stack(true_controls)
    true_states_tensor = true_states_tensor.permute(1, 0, 2)
    true_controls_tensor = true_controls_tensor.permute(1, 0, 2)
    return violation_mask, cte_mins, cte_maxs, true_states_tensor, true_controls_tensor

def calculate_action(states):
    """
    Calculate actions for a trajectory of states.
    
    Args:
        states: tensor of shape (batch_size, K, 2) where each state is (d, he)
    Returns:
        actions: tensor of shape (batch_size, K, 1)
    """
    batch_size, K, _ = states.shape
    d = states[..., 0]
    he = states[..., 1]
    
    # actions based on empirical PID controls
    actions = -10.0 * (d/100) - 3.0 * he
    actions = torch.clamp(actions, min=-30, max=30)
    # normalize actions
    actions = convert_action(actions, 'normalize')
    actions = actions.view(batch_size, K, 1)
    return actions