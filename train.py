import time
import torch
import numpy as np

from spvt.args import get_args
from spvt.utils import prepare_train, create_latent_ptb_tensor, save_ckpt
from spvt.scripts import KScheduler, HardExampleBuffer, RandomExampleSet, BatchSampler, adjust_hyperparameters
from spvt.loss import compute_loss
from spvt.logger import EpochInfoPrinter


def train(args):
    model_ori, model, optimizer, lr_scheduler, dataloader, logger, loss_tracker, ckpt = prepare_train(args)
    HARD_SET = HardExampleBuffer(max_size=args.buffer_size)
    K_SCHEDULER = KScheduler(args.k_min, args.k_max, 3*(args.k_max-args.k_min))
    BATCH_SAMPLER = BatchSampler(args.batch_fraction)
    PRINTER = EpochInfoPrinter()
    init_states = []
    for state_batch in dataloader:
        init_states.append(state_batch)
    init_states = torch.cat(init_states, dim=0)
    RANDOM_SET = RandomExampleSet(init_states, device=args.device)
    
    # Training loop
    model.train()
    best_safe_loss = float('inf')
    starting_epoch = 1 if ckpt is None else ckpt['epoch'] + 1

    for epoch in range(starting_epoch, args.num_epochs+1):
        K_SCHEDULER.update(epoch)
        curr_K = K_SCHEDULER.get_K()
        PRINTER.print_epoch_header(epoch, args.num_epochs, curr_K)
        
        loss_tracker.reset_epoch_losses()
        start_time = time.time()
        states_added_to_buffer = 0
        batch_size = args.batch_size
        n_batches = (len(RANDOM_SET) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(RANDOM_SET))
            indices = list(range(start_idx, end_idx))
            curr_batch_size = end_idx - start_idx
            state_batch = RANDOM_SET[indices]
            if isinstance(state_batch, list):
                state_batch = torch.stack(state_batch)
            if len(state_batch.shape) == 1:
                state_batch = state_batch.unsqueeze(0)
            state_batch = state_batch.to(args.device)
            state_batch = BATCH_SAMPLER.get_train_batch(args, state_batch, HARD_SET)
            latent_ptb_tensor = create_latent_ptb_tensor(args, model_ori, state_batch.size(0))

            optimizer.zero_grad()
            loss, pred_loss, safety_loss, bound_loss, reg_loss, state_lp_loss, latent_lp_loss, HARD_SET, added_batch_indices = compute_loss(
                args, model, state_batch, latent_ptb_tensor, curr_K, HARD_SET
            )
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            latest_batch_loss = {
                'total_loss': loss.item(),
                'pred_loss': pred_loss.item(),
                'safety_loss': safety_loss.item(),
                'bound_loss': bound_loss.item(),
                'reg_loss': reg_loss.item(),
                'state_lp_loss': state_lp_loss.item(),
                'latent_lp_loss': latent_lp_loss.item()
            }
            if added_batch_indices:
                added_dataset_indices = [indices[idx] for idx in added_batch_indices if idx < len(indices)]
                if added_dataset_indices:
                    removed_states = RANDOM_SET.remove_states(added_dataset_indices)
                    n_removed = len(removed_states)
                    RANDOM_SET.generate_and_add_states(n_removed)
                    states_added_to_buffer += n_removed
            
            losses = {
                'total_loss': loss,
                'pred_loss': pred_loss,
                'safety_loss': safety_loss,
                'bound_loss': bound_loss,
                'reg_loss': reg_loss,
                'state_lp_loss': state_lp_loss,
                'latent_lp_loss': latent_lp_loss
            }
            loss_tracker.update_batch_losses(losses)

        epoch_time = time.time() - start_time
        epoch_safety_loss, epoch_pred_acc_loss = loss_tracker.log_epoch_summary(epoch, start_time, print_summary=False)
        loss_metrics = {
            'Total loss': loss_tracker.epoch_losses['loss/total'] / n_batches,
            'Prediction loss': loss_tracker.epoch_losses['loss/pred'] / n_batches,
            'Bound gap loss': loss_tracker.epoch_losses['loss/bound'] / n_batches,
            'Safety loss': loss_tracker.epoch_losses['loss/safety'] / n_batches,
            'Regularization loss': loss_tracker.epoch_losses['loss/reg'] / n_batches,
            'LP state loss': loss_tracker.epoch_losses['loss/state_lp'] / n_batches,
            'LP latent loss': loss_tracker.epoch_losses['loss/latent_lp'] / n_batches
        }
        buffer_metrics = {
            'States added this epoch': states_added_to_buffer,
            'Buffer size': len(HARD_SET),
            'Buffer violation rate': HARD_SET.get_violation_rate(),
            'Dataset size': len(RANDOM_SET)
        }
        param_metrics = {
            'Learning rate': optimizer.param_groups[0]['lr'],
            'Batch fraction': BATCH_SAMPLER.batch_fraction,
            'Dynamic lambda': args.dynamic_lambda
        }
        PRINTER.log_training_summary(
            epoch, 
            args.num_epochs, 
            curr_K,
            loss_metrics, 
            buffer_metrics, 
            param_metrics, 
            epoch_time
        )
        tb_metrics = {
            'buffer/states_added': states_added_to_buffer,
            'buffer/size': len(HARD_SET),
            'buffer/violation_rate': HARD_SET.get_violation_rate(),
            'dataset/size': len(RANDOM_SET),
            'params/K': curr_K,
            'params/lr': optimizer.param_groups[0]['lr'],
            'params/zeta': BATCH_SAMPLER.batch_fraction,
            'params/alpha': args.dynamic_lambda,
        }
        logger.log_metrics(tb_metrics, epoch)
        lr_scheduler.step()
        args.batch_fraction, args.dynamic_lambda = adjust_hyperparameters(args, epoch_pred_acc_loss)
        BATCH_SAMPLER.set_batch_fraction(args.batch_fraction)

        if epoch % 5 == 0:
            save_ckpt(args, model_ori, model, optimizer, epoch)
        if epoch_safety_loss < best_safe_loss and epoch > (args.num_epochs / 2):
            best_safe_loss = epoch_safety_loss
            save_ckpt(args, model_ori, model, optimizer, epoch)
        

if __name__ == '__main__':
    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    from spvt.args import print_args_by_category
    print_args_by_category(args)
    
    train(args)