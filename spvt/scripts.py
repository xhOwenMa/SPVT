import heapq
from dataclasses import dataclass
from typing import List, Tuple
import torch
import random
import math


def adjust_hyperparameters(args, epoch_pred_acc_loss):
    """
    Adjust hyperparameters based on training progress.
    
    Returns:
        Tuple of (batch_fraction, dynamic_lambda)
    """
    batch_fraction = args.batch_fraction
    dynamic_lambda = args.dynamic_lambda
    
    if epoch_pred_acc_loss > args.target_acc_loss:
        dynamic_lambda = min(0.9, dynamic_lambda + args.dynamic_step)
        batch_fraction = min(0.9, batch_fraction + args.dynamic_step)
    else:
        dynamic_lambda = max(0.1, dynamic_lambda - args.dynamic_step)
        batch_fraction = max(0.1, batch_fraction - args.dynamic_step)
    
    return batch_fraction, dynamic_lambda

class BatchSampler:
    def __init__(self, batch_fraction: float = 0.5):
        self.batch_fraction = batch_fraction
    
    def set_batch_fraction(self, fraction: float):
        """Allows dynamic adjustment of batch fraction during training"""
        assert 0 < fraction <= 1.0, "Batch fraction must be between 0 and 1"
        self.batch_fraction = fraction
    
    def get_train_batch(self, args, state_batch, buffer):
        batch_size = state_batch.size(0)
        if len(buffer) > batch_size:
            sampled_X = state_batch
        else:
            num_samples = int(self.batch_fraction * batch_size)
            selected_indices = random.sample(range(len(state_batch)), num_samples)
            sampled_X = state_batch[selected_indices]
            buffer_X, _ = buffer.get_training_batch(batch_size - num_samples)
            sampled_X = torch.cat([sampled_X, buffer_X], dim=0)
        return sampled_X.to(args.device)


@dataclass
class LossExample:
    """
    Data class for storing examples in the counter-example buffer.
    
    Attributes:
        state: State tensor (cte, he)
        loss: Loss value associated with this example
        violated: Boolean indicating if this example violated safety constraints
    """
    state: torch.Tensor
    loss: float
    violated: bool

    def __lt__(self, other):
        """
        Custom comparison for priority queue ordering.
        Higher loss examples have higher priority (are "greater").
        """
        return self.loss > other.loss

class CounterExampleBuffer:
    """
    Buffer that stores hard examples and safety violations for improved training.
    
    This class implements a priority queue of examples that were difficult
    for the model, either because they had high loss values or they resulted
    in safety violations. These examples are used to augment training batches.
    
    The implementation uses a min-heap where the smallest (lowest loss) example
    is always at the top, allowing efficient replacement when the buffer is full.
    """
    def __init__(self, max_size: int = 4000):
        """
        Initialize the counter example buffer.
        
        Args:
            max_size: Maximum number of examples to store (default: 4000)
        """
        self.max_size = max_size
        self.buffer: List[LossExample] = []  # Implemented as a min-heap
        self.violation_count = 0  # Track number of safety violation examples
        self.added = 0  # Counter for newly added examples in each call

    def add_examples(self, 
                states: torch.Tensor, 
                losses: torch.Tensor,
                violation_mask: torch.Tensor,
                worst_percentile: float = 10.0):
        """
        Add examples to the buffer with violation prioritization.
        
        Args:
            states: Tensor of shape (batch_size, state_dim) containing states
            losses: Tensor of shape (batch_size,) containing loss values
            violation_mask: Bool tensor of shape (batch_size, K) indicating 
                           which examples violated safety constraints
            worst_percentile: Percentage of worst examples to add (default: 10%)
        """
        self.added = 0
        self._add_worst_examples(states, losses, violation_mask, worst_percentile)

    def _add_worst_examples(self, states, losses, violation_mask, worst_percentile):
        """
        Add worst examples based on loss values.
        
        Args:
            states: Tensor of states
            losses: Tensor of loss values
            violation_mask: Boolean mask of safety violations
            worst_percentile: Percentage of worst examples to add
        """
        # Calculate how many examples to add based on percentile
        n_examples = max(1, int(len(losses) * (worst_percentile / 100)))
        
        # Sort losses in descending order and get indices
        _, sorted_indices = torch.sort(losses, descending=True)
        
        # Add examples until we reach the target count
        for idx in sorted_indices:
            if self.added >= n_examples:
                break
            else:
                self._add_single_example(
                    state=states[idx],
                    loss=losses[idx].item(),
                    violated=violation_mask[idx].any().item()
                )

    def _add_single_example(self, state, loss, violated):
        """
        Add a single example to the buffer, maintaining heap property.
        
        Args:
            state: State tensor
            loss: Loss value
            violated: Whether this example violated safety constraints
        """
        # Create detached copy of state tensor
        processed_state = state.squeeze().clone().detach()
        
        # Create example object
        example = LossExample(
            state=processed_state,
            loss=loss,
            violated=violated
        )
        
        # If buffer isn't full, add example
        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, example)
            self.added += 1
            if violated:
                self.violation_count += 1
        else:
            # If buffer is full but new example has higher loss than minimum
            if loss > self.buffer[0].loss:
                self.added += 1
                # Update violation count
                if self.buffer[0].violated and not violated:
                    self.violation_count -= 1
                elif not self.buffer[0].violated and violated:
                    self.violation_count += 1
                # Replace the minimum loss example
                heapq.heapreplace(self.buffer, example)

    def get_violation_rate(self):
        """
        Get percentage of buffer containing safety violation examples.
        
        Returns:
            Float representing violation rate (0.0 to 1.0)
        """
        if not self.buffer:
            return 0.0
        return self.violation_count / len(self.buffer)
    
    def get_training_batch(self, batch_size: int, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of examples for training using importance-weighted sampling.
        
        Each example is assigned a weight based on its loss:
            w_i = exp(alpha * loss_i)
        
        Higher alpha values emphasize higher-loss examples more strongly.
        
        Args:
            batch_size: Number of examples to sample
            alpha: Importance weighting factor (default: 1.0)
            
        Returns:
            Tuple of (states, losses)
            
        Raises:
            ValueError: If buffer is empty
        """
        if not self.buffer:
            raise ValueError("Buffer is empty")
            
        # Sample at most batch_size examples
        n_samples = min(batch_size, len(self.buffer))
        
        # Calculate importance weights
        weights = [math.exp(alpha * example.loss) for example in self.buffer]
        
        # Sample with replacement according to weights
        sampled_indices = random.choices(range(len(self.buffer)), weights=weights, k=n_samples)
        
        # Collect states and losses
        states = torch.stack([self.buffer[idx].state for idx in sampled_indices])
        losses = torch.tensor([self.buffer[idx].loss for idx in sampled_indices])
        
        return states, losses
    
    def __len__(self):
        """Return the current number of examples in the buffer."""
        return len(self.buffer)


class KScheduler:
    def __init__(self, min = 5, max = 20, ramp_epochs = 30):
        self.min_K = min
        self.max_K = max
        self.K_ramp_epochs = ramp_epochs
        self.current_K = min
        
    def update(self, epoch: int):
        """Update K based on training progress"""
        if epoch < self.K_ramp_epochs:
            # Linear increase during ramp-up phase
            self.current_K = min(
                self.max_K,
                self.min_K + int((epoch / self.K_ramp_epochs) * (self.max_K - self.min_K))
            )
        else:
            self.current_K = self.max_K
            
    def get_K(self):
        return self.current_K