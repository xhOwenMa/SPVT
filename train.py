import time
import torch

from spvt.args import get_args
from spvt.utils import prepare_train, create_latent_ptb_tensor, save_ckpt
from spvt.scripts import KScheduler, CounterExampleBuffer, BatchSampler, adjust_hyperparameters
from spvt.loss import compute_loss


def train(args):
    model_ori, model, optimizer, lr_scheduler, dataloader, logger, loss_tracker, ckpt = prepare_train(args)
    BUFFER = CounterExampleBuffer(max_size=args.buffer_size)
    K_SCHEDULER = KScheduler(args.k_min, args.k_max, 3*(args.k_max-args.k_min))
    BATCH_SAMPLER = BatchSampler(args.batch_fraction)

    # Training loop
    model.train()
    best_safe_loss = float('inf')
    starting_epoch = 1 if ckpt is None else ckpt['epoch'] + 1

    for epoch in range(starting_epoch, args.num_epochs+1):
        K_SCHEDULER.update(epoch)
        curr_K = K_SCHEDULER.get_K()
        print(f"\nEpoch {epoch}/{args.num_epochs} with K={curr_K}")
        
        loss_tracker.reset_epoch_losses()
        start_time = time.time()

        for i, state_batch in enumerate(dataloader):
            batch_size = state_batch.size(0)
            state_batch = state_batch.to(args.device)
            state_batch = BATCH_SAMPLER.get_train_batch(args, state_batch, BUFFER)
            latent_ptb_tensor = create_latent_ptb_tensor(args, model_ori, batch_size)

            optimizer.zero_grad()
            loss, pred_loss, safety_loss, bound_loss, reg_loss, state_lp_loss, latent_lp_loss, BUFFER = compute_loss(args, model, state_batch, latent_ptb_tensor, curr_K, BUFFER)
            
            losses = {
                'total_loss': loss,
                'pred_loss': pred_loss,
                'safety_loss': safety_loss,
                'bound_loss': bound_loss,
                'reg_loss': reg_loss,
                'state_lp_loss': state_lp_loss,
                'latent_lp_loss': latent_lp_loss
            }
            
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            loss_tracker.update_batch_losses(losses)
            loss_tracker.log_batch_info(i, losses)

        epoch_info = {
            'params/K': curr_K,
            'params/lr': optimizer.param_groups[0]['lr'],
            'params/zeta': BATCH_SAMPLER.batch_fraction,
            'params/alpha': args.dynamic_lambda,
        }
        logger.log_metrics(epoch_info, epoch)

        lr_scheduler.step()

        epoch_safety_loss, epoch_pred_acc_loss = loss_tracker.log_epoch_summary(epoch, start_time)
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
    
    from spvt.args import print_args_by_category
    print_args_by_category(args)
    
    train(args)