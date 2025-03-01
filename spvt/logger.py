from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any
import os
from argparse import Namespace

import time
from colorama import Fore, Style, init

init()

class EpochInfoPrinter:
    def __init__(self):
        """Initialize the pretty logger for training outputs."""
        self.timer = 0.0
        
    def print_epoch_header(self, epoch, total_epochs, k_value):
        """Print a formatted header for the epoch."""
        box_width = 80
        separator = "=" * box_width
        
        print("\n" + separator)
        print(f"{Fore.GREEN}EPOCH {epoch}/{total_epochs}{Style.RESET_ALL} - K={k_value}")
        print(separator)
        
    def print_metrics_by_category(self, metrics_dict, box_width=80):
        """
        Print metrics organized by categories in a nicely formatted box.
        
        Args:
            metrics_dict: Dictionary of dictionaries, where the top-level keys are category names
                          and values are dictionaries of metrics in that category
            box_width: Width of the box
        """
        separator = "-" * box_width
        
        for category, metrics in metrics_dict.items():
            # Skip empty categories
            if not metrics:
                continue
                
            # Print category header
            print(f"\n{Fore.YELLOW}{category}{Style.RESET_ALL}")
            print(separator)
            
            # Find the longest metric name for alignment
            max_name_length = max((len(name) for name in metrics.keys()), default=0)
            
            # Print each metric with aligned values
            for name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {name:<{max_name_length+3}}: {value:.6f}")
                else:
                    print(f"  {name:<{max_name_length+3}}: {value}")
        
    def log_training_summary(self, epoch, total_epochs, k_value, losses, buffer_stats, params, epoch_time):
        """
        Log a comprehensive training summary after each epoch.
        
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            k_value: Current K value
            losses: Dictionary of loss values
            buffer_stats: Dictionary of buffer statistics
            params: Dictionary of hyperparameters
            epoch_time: Time taken for the epoch
        """
        # Update total training time
        self.timer += epoch_time
        hrs, rem = divmod(self.timer, 3600)
        mins, secs = divmod(rem, 60)
        
        # Time metrics
        time_metrics = {
            "Epoch time": f"{int(epoch_time // 60)}m:{int(epoch_time % 60)}s",
            "Total training time": f"{int(hrs):02d}h:{int(mins):02d}m:{int(secs):02d}s"
        }
        
        # Organize metrics by category
        metrics_by_category = {
            "LOSS METRICS": losses,
            "BUFFER STATISTICS": buffer_stats,
            "HYPERPARAMETERS": params,
            "TIME STATISTICS": time_metrics
        }
        
        # Print metrics by category
        self.print_metrics_by_category(metrics_by_category)
        
        # Print footer
        print("\n" + "=" * 80)

class Logger:
    def __init__(self, log_dir: str):
        """Initialize the logger."""
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics at once."""
        if step is None:
            step = self.step
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
        self.step += 1

    def log_scalar(self, name: str, value: float, step: int = None):
        """Log a single scalar value."""
        if step is None:
            step = self.step
        self.writer.add_scalar(name, value, step)

    def log_image(self, name: str, image, step: int = None):
        """Log an image tensor."""
        if step is None:
            step = self.step
        self.writer.add_image(name, image, step)

    def log_text(self, name: str, text: str, step: int = None):
        """Log text content."""
        if step is None:
            step = self.step
        self.writer.add_text(name, text, step)

    def log_args(self, args: Namespace):
        """Log command line arguments."""
        text = "### Training Configuration ###\n\n"
        categories = {
            "Run ID": ["run_id"],
            "Hyperparameters": ["latent_dim", "k_min", "k_max", "beta1", "beta2", 
                              "num_epochs", "batch_size", "lr"],
            "Training Settings": ["safety_loss_type", "buffer_criteria", "buffer_size",
                                "bound_method", "lambda_safety", "lambda_bound", "pred_loss_step",
                                "lambda_accuracy", "lambda_reg", "lambda_lp",
                                "dynamic_lambda", "target_acc_loss", "dynamic_step",
                                "batch_fraction"],
            "Other": ["ckpt", "device", "seed"]
        }
        
        for category, arg_names in categories.items():
            text += f"\n## {category}\n"
            for arg_name in arg_names:
                if hasattr(args, arg_name):
                    value = getattr(args, arg_name)
                    text += f"* {arg_name}: `{value}`\n"
        
        self.log_text("training_config", text, 0)

    def close(self):
        """Close the tensorboard writer."""
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LossTracker:
    def __init__(self, logger, total_batches):
        self.logger = logger
        self.total_batches = total_batches
        self.timer = 0.0
        self.reset_epoch_losses()

    def reset_epoch_losses(self):
        self.epoch_losses = {
            'loss/pred': 0,
            'loss/bound': 0,
            'loss/safety': 0,
            'loss/reg': 0,
            'loss/state_lp': 0,
            'loss/latent_lp': 0,
            'loss/total': 0
        }

    def update_batch_losses(self, losses):
        key_mapping = {
            'pred_loss': 'loss/pred',
            'bound_loss': 'loss/bound',
            'safety_loss': 'loss/safety',
            'reg_loss': 'loss/reg',
            'state_lp_loss': 'loss/state_lp',
            'latent_lp_loss': 'loss/latent_lp',
            'total_loss': 'loss/total'
        }
        for input_key, tb_key in key_mapping.items():
            if input_key in losses:
                self.epoch_losses[tb_key] += losses[input_key].item()

    def log_batch_info(self, batch_idx, losses):
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx+1}/{self.total_batches}:")
            print(
                f"Loss: {losses['total_loss'].item():.4f} | "
                f"Pred: {losses['pred_loss'].item():.4f} | "
                f"Safety: {losses['safety_loss'].item():.4f} | "
                f"Bound Gap: {losses['bound_loss'].item():.4f}")
            print(f"Regularizations: "
                    f"Reg: {losses['reg_loss'].item():.4f} | "
                    f"LP State: {losses['state_lp_loss'].item():.4f} | "
                    f"LP Latent: {losses['latent_lp_loss'].item():.4f}")

    def log_epoch_summary(self, epoch, start_time, print_summary=True):
        avg_losses = {k: v / self.total_batches for k, v in self.epoch_losses.items()}

        if print_summary:
            print(f"\nEpoch {epoch} Summary:")
            print(f"Total loss: {avg_losses['loss/total']:.4f}")
            print(f"Prediction loss: {avg_losses['loss/pred']:.4f}")
            print(f"Bound gap loss: {avg_losses['loss/bound']:.4f}")
            print(f"Batch safety loss: {avg_losses['loss/safety']:.4f}")
            print(f"reg loss: {avg_losses['loss/reg']:.4f}")
            print(f"lp state loss: {avg_losses['loss/state_lp']:.4f}")
            print(f"lp latent loss: {avg_losses['loss/latent_lp']:.4f}")

            epoch_time = time.time() - start_time
            self.timer += epoch_time
            self._log_time(epoch_time)
            
        self.logger.log_metrics(avg_losses, step=epoch)
        
        return avg_losses['loss/safety'], avg_losses['loss/pred']

    def _log_time(self, epoch_time):
        epoch_mins, epoch_secs = epoch_time // 60, epoch_time % 60
        print(f'Epoch time: {int(epoch_mins)}m:{int(epoch_secs)}s')
        
        hrs, rem = divmod(self.timer, 3600)
        mins, secs = divmod(rem, 60)
        print(f'Total training time: {int(hrs):02d}h:{int(mins):02d}m:{int(secs):02d}s')

