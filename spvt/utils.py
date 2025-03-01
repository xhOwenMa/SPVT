import sys
from pathlib import Path
import numpy as np
import os
import math

current_dir = Path(__file__).parent
parent_dir = str(current_dir.parent)

sys.path.insert(0, parent_dir)
sys.path.insert(0, str(current_dir))

from model.model import get_model
from logger import Logger, LossTracker

import torch
from torch.optim import lr_scheduler

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import sync_params
from auto_LiRPA.bound_ops import *


def convert_action(actions, direction):
    """
    convert (normalize/unnormalize) actions
    """
    if direction == 'normalize':
        return (actions + 30) / 60 * 2 - 1
    elif direction == 'unnormalize':
        return (actions + 1) / 2 * 60 - 30
    
def dynamics(cte, he, phi, dt=0.05, v=5, L=2.5):
    """ dynamics model
        Args:
            cte: current crosstrack error (centimeters)
            he: current heading error (degrees)
            phi: steering angle (controller output) (normalized)
            -------------------------------
            dt: time step (seconds)
            v: speed (m/s)
            L: distance between front and back wheels (meters)
    """
    he = he.view(-1)
    phi = phi.view(-1) 
    cte = cte.view(-1)

    he_rad = torch.deg2rad(he)
    cte_m = cte / 100
    phi = convert_action(phi, 'unnormalize')
    phi_rad = torch.deg2rad(phi)

    he_next_rad = he_rad + v * torch.tan(phi_rad) * dt / L
    he_next_rad = he_next_rad % (2 * torch.pi)
    mask = he_next_rad > torch.pi
    he_next_rad = torch.where(mask, he_next_rad - 2 * torch.pi, he_next_rad)
    he_next = torch.rad2deg(he_next_rad)

    cte_next_m = cte_m + v * torch.sin(he_rad) * dt
    cte_next = cte_next_m * 100

    return cte_next, he_next
    

def prepare_train(args):
    """
    prepare training: create logger, models, optimizers, dataloaders, etc.
    """
    # 1. create logger and log args
    logger = Logger(f'{current_dir}/{args.tb_path}')
    logger.log_args(args)

    # 2. create models
    model_ori = get_model()
    model_ori.generator.load_state_dict(torch.load(args.gen_ckpt, map_location=args.device)['generator_state_dict'])
    model_ori.controller.load_state_dict(torch.load(args.control_ckpt, map_location=args.device))
    if args.ckpt is not None:
        ckpt = torch.load(f'veri_train_logs/{args.ckpt}/ckpts', map_location=args.device)
        model_ori.load_state_dict(ckpt['model_state_dict'])
    else:
        ckpt = None
    # wrap the model with auto_LiRPA
    model_ori.eval()
    model = create_bounded_module(args, model_ori)
    model.train()

    # 3. create optimizer
    controller_params = [p for name, p in model.named_parameters() if 'controller' in name and p.requires_grad]
    optimizer = torch.optim.Adam(controller_params, lr=args.lr, betas=(args.beta1, args.beta2))
    # learning rate scheduler
    lr_scheduler = get_lr_scheduler(optimizer, num_epochs=args.num_epochs, eta_min=args.lr_end)

    # 4. create dataloader
    try:
        state_data = torch.load(f'{parent_dir}/data/{args.exp}_states.pt')
        dataloader = torch.utils.data.DataLoader(
            state_data, 
            batch_size=args.batch_size, 
            shuffle=True,
            drop_last=False
        )
    except FileNotFoundError:
        print()
        print('\033[91m===================STATE DATA NOT FOUND===================\033[0m')
        print('randomly generating 1000 states...')
        print('\033[93mWARNING: this is not enough data for training\033[0m')
        cte = torch.rand(1000) * 100 - 50
        he = torch.rand(1000) * 60 - 30
        states = torch.stack([cte, he], dim=1)
        dataloader = torch.utils.data.DataLoader(
            states, 
            batch_size=args.batch_size, 
            shuffle=True,
            drop_last=False
        )

    # 5. create loss tracker
    loss_tracker = LossTracker(logger, len(dataloader))

    return model_ori, model, optimizer, lr_scheduler, dataloader, logger, loss_tracker, ckpt


def create_bounded_module(args, model):
    """
    Create a BoundedModule from a model.
    """
    for name, param in model.named_parameters():
        if name.startswith('generator'):
            param.requires_grad = False

    state_input = torch.randn(1, 2).to(args.device)
    latent_input = torch.randn(1, 10).to(args.device)
    lirpa_model = BoundedModule(model, (latent_input, state_input), device=args.device)
    
    for name, param in lirpa_model.named_parameters():
        if name.startswith('generator'):
            param.requires_grad = False
    # print to check if the parameters are correctly set
    for name, param in lirpa_model.named_parameters():
        print(name, param.requires_grad)

    return lirpa_model

def create_latent_ptb_tensor(args, model, batch_size):
    """
    Create a perturbation tensor for the latent input.
    """
    z = model.generator.sample_z(batch_size, args.latent_dim, device=args.device)
    latent_max = 0.8 * torch.ones(batch_size, 10).to(args.device)
    latent_min = -0.8 * torch.ones(batch_size, 10).to(args.device)
    ptb = PerturbationLpNorm(norm=np.inf, x_L=latent_min, x_U=latent_max)
    latent_ptb_tensor = BoundedTensor(z, ptb)
    return latent_ptb_tensor

def get_lr_scheduler(optimizer, num_epochs, eta_min=1e-6):
    """
    Create and return a learning rate scheduler.
    TODO: implement other learning rate schedulers, maybe?
    """
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    return scheduler

def save_ckpt(args, model_ori, model, optimizer, epoch):
    """
    Save a checkpoint of the model and optimizer state.
    """
    state_dict = sync_params(model_ori, model)
    ckpt = {
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    os.makedirs(f'{current_dir}/{args.ckpt_path}', exist_ok=True)
    torch.save(ckpt, f'{current_dir}/{args.ckpt_path}/checkpoint_epoch_{epoch}.pth')


def get_params(model):
    weights = []
    biases = []
    for p in model.named_parameters():
        if 'weight' in p[0]:
            if p[0].startswith('generator'):
                print('Skipping generator weights {}'.format(p[0]))
                # weights.append(p)
            else:
                weights.append(p)
        elif 'bias' in p[0]:
            biases.append(p)
        else:
            print('Skipping parameter {}'.format(p[0]))
    return weights, biases

def ibp_init(model_ori, model):
    print('Reinitializing weights using IBP init')
    weights, biases = get_params(model_ori)
    for i in range(len(weights)-1):
        if weights[i][1].ndim == 1:
            continue
        weight = weights[i][1]
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(2 * math.pi / (fan_in**2))
        std_before = weight.std().item()
        torch.nn.init.normal_(weight, mean=0, std=std)
        print(f'Reinitialize {weights[i][0]}, std before {std_before:.5f}, std now {weight.std():.5f}')
    for node in model._modules.values():
        if isinstance(node, BoundConv) or isinstance(node, BoundLinear):
            if len(node.inputs[0].inputs) > 0 and isinstance(node.inputs[0].inputs[0], BoundAdd):
                print(f'Adjust weights for node {node.name} due to residual connection')
                node.inputs[1].param.data /= 2