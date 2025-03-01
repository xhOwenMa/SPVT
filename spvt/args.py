import argparse
from datetime import datetime

def get_run_id(parser):
    test = True
    if test:
        timestamp = 'test'
    else:
        timestamp = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    parser.add_argument('--run_id', type=str, default=f'logs/{timestamp}', help='run id')
    parser.add_argument('--tb_path', type=str, default=f'logs/{timestamp}/runs', help='tensorboard path')
    parser.add_argument('--ckpt_path', type=str, default=f'logs/{timestamp}/ckpts', help='checkpoint path')
    return parser

def get_hyperparameters(parser):
    parser.add_argument('--latent_dim', type=int, default=10, help='latent dimension of the generator')

    parser.add_argument('--k_min', type=int, default=4, help='k_min for sim trajectory')
    parser.add_argument('--k_max', type=int, default=10, help='k_max for sim trajectory')

    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=8e-5, help='learning rate')
    parser.add_argument('--lr_end', type=float, default=1e-5, help='minimum learning rate')
    return parser
    
def get_train_args(parser):
    parser.add_argument('--pred_loss_step', type=int, default=1, help='step for prediction loss')
    parser.add_argument('--safety_loss_type', type=str, default='single_step', help='cumulative or barrier or single_step')
    parser.add_argument('--buffer_criteria', type=str, default='bound_gap', help='safety_loss or bound_gap')
    parser.add_argument('--buffer_size', type=int, default=4000, help='buffer size')
    parser.add_argument('--bound_method', type=str, default='backward', help='backward or IBP+backward or IBP')

    # loss weights
    parser.add_argument('--lambda_safety', type=float, default=0.0, help='safety loss weight')
    parser.add_argument('--lambda_bound', type=float, default=0.0, help='bound loss weight')
    parser.add_argument('--lambda_accuracy', type=float, default=0.1, help='accuracy loss weight')
    parser.add_argument('--lambda_reg', type=float, default=0.0, help='ReLU regularization loss weight')
    parser.add_argument('--lambda_lp', type=float, default=0.0, help='Lp regularization loss weight')

    # dynamic loss weight: balance safety and accuracy
    parser.add_argument('--dynamic_lambda', type=float, default=0.5, help='dynamic loss weight to balance safety and accuracy: higher value means more emphasis on accuracy')
    parser.add_argument('--target_acc_loss', type=float, default=0.04, help='target accuracy loss')
    parser.add_argument('--dynamic_step', type=float, default=0.05, help='step to adjust dynamic loss weight')

    # batch fraction: balance random samples and buffer hard samples
    parser.add_argument('--batch_fraction', type=float, default=0.5, help='buffer weight: higher value means more emphasis on random samples')
    return parser

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Veri')
    parser = get_run_id(parser)
    parser = get_hyperparameters(parser)
    parser = get_train_args(parser)

    # TODO: pretrained checkpoints can be found based on 'exp', but this is not implemented yet
    parser.add_argument('--gen_ckpt', type=str, default='model/pretrained_ckpts/mlp_generator.pth', help='pretrained generator checkpoint')
    parser.add_argument('--control_ckpt', type=str, default='model/pretrained_ckpts/pid_controller.pth', help='pretrained controller checkpoint')
    parser.add_argument('--ckpt', type=str, default=None, help='spvt checkpoint file id')  # NOTE: this is really not useful...
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    # TODO: if running xplane experiments, need to adjust dynamics and action ranges; need to make this automatic
    parser.add_argument('--exp', type=str, default='carla', help='carla or xplane')

    args = parser.parse_args()

    if args.buffer_criteria == 'bound_gap':
        args.lambda_bound = 1.0
    elif args.buffer_criteria == 'safety_loss':
        args.lambda_safety = 1.0
    args.lambda_accuracy = 1.0 / args.pred_loss_step

    return args


def print_args_by_category(args):
    """
    Print arguments grouped by category with clear visual separation.
    
    Args:
        args: Parsed arguments from get_args()
    """
    import torch

    separator = "=" * 80
    
    # Run ID parameters
    print(separator)
    print("RUN ID PARAMETERS:")
    print(f"  run_id             : {args.run_id}")
    print(f"  tb_path            : {args.tb_path}")
    print(f"  ckpt_path          : {args.ckpt_path}")
    print()
    
    # Hyperparameters
    print(separator)
    print("HYPERPARAMETERS:")
    print(f"  latent_dim         : {args.latent_dim}")
    print(f"  k_min              : {args.k_min}")
    print(f"  k_max              : {args.k_max}")
    print(f"  beta1              : {args.beta1}")
    print(f"  beta2              : {args.beta2}")
    print(f"  num_epochs         : {args.num_epochs}")
    print(f"  batch_size         : {args.batch_size}")
    print(f"  lr                 : {args.lr}")
    print(f"  lr_end             : {args.lr_end}")
    print()
    
    # Training parameters
    print(separator)
    print("TRAINING PARAMETERS:")
    print(f"  pred_loss_step     : {args.pred_loss_step}")
    print(f"  safety_loss_type   : {args.safety_loss_type}")
    print(f"  buffer_criteria    : {args.buffer_criteria}")
    print(f"  buffer_size        : {args.buffer_size}")
    print(f"  bound_method       : {args.bound_method}")
    print()
    
    # Loss weights
    print(separator)
    print("LOSS WEIGHTS:")
    print(f"  lambda_safety      : {args.lambda_safety}")
    print(f"  lambda_bound       : {args.lambda_bound}")
    print(f"  lambda_accuracy    : {args.lambda_accuracy}")
    print(f"  lambda_reg         : {args.lambda_reg}")
    print(f"  lambda_lp          : {args.lambda_lp}")
    print(f"  dynamic_lambda     : {args.dynamic_lambda}")
    print(f"  target_acc_loss    : {args.target_acc_loss}")
    print(f"  dynamic_step       : {args.dynamic_step}")
    print(f"  batch_fraction     : {args.batch_fraction}")
    print()
    
    # Model parameters
    print(separator)
    print("MODEL PARAMETERS:")
    print(f"  gen_ckpt           : {args.gen_ckpt}")
    print(f"  control_ckpt       : {args.control_ckpt}")
    print(f"  ckpt               : {args.ckpt}")
    print()
    
    # Other parameters
    print(separator)
    print("OTHER PARAMETERS:")
    print(f"  device             : {args.device} --- {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"  seed               : {args.seed}")
    print(f"  exp        : {args.exp}")
    print(separator)
    print()
