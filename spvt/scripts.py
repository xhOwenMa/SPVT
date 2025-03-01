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
        if len(buffer) < batch_size:
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
    Data class for storing examples in the hard example buffer.
    
    Attributes:
        state: State tensor (cte, he)
        loss: float
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

class HardExampleBuffer:
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
        Initialize the hard example buffer.
        
        Args:
            max_size: Maximum number of examples to store (default: 4000)
        """
        self.max_size = max_size
        self.buffer: List[LossExample] = []  # a min-heap
        self.violation_count = 0
        self.added = 0
        self.added_indices = []
        self.state_hashes = set()
        
    def _hash_state(self, state):
        """Convert a state tensor to a hashable object."""
        return hash(state.detach().cpu().numpy().tobytes())
        
    def contains_state(self, state):
        """Check if a state is in the buffer."""
        return self._hash_state(state) in self.state_hashes

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
                           
        Returns:
            List of indices of examples that were added to the buffer
        """
        self.added = 0
        self.added_indices = []  # Reset the added indices list
        self._add_worst_examples(states, losses, violation_mask, worst_percentile)
        return self.added_indices  # Return indices of added examples

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
            
            # Get the state and its hash
            state = states[idx]
            state_hash = self._hash_state(state)
            
            # Only add if not already in buffer
            if state_hash not in self.state_hashes:
                added = self._add_single_example(
                    state=state,
                    loss=losses[idx].item(),
                    violated=violation_mask[idx].any().item(),
                    state_hash=state_hash
                )
                
                if added:
                    # Store the index of the added example
                    self.added_indices.append(idx.item())

    def _add_single_example(self, state, loss, violated, state_hash):
        """
        Add a single example to the buffer, maintaining heap property.
        
        Args:
            state: State tensor
            loss: Loss value
            violated: Whether this example violated safety constraints
            state_hash: Hash of the state for efficient lookup
            
        Returns:
            Boolean indicating whether the example was added
        """
        processed_state = state.squeeze().clone().detach()
        example = LossExample(
            state=processed_state,
            loss=loss,
            violated=violated
        )
        if len(self.buffer) < self.max_size:
            heapq.heappush(self.buffer, example)
            self.added += 1
            if violated:
                self.violation_count += 1
            self.state_hashes.add(state_hash)
            
            return True
        else:
            if loss > self.buffer[0].loss:
                self.added += 1
                if self.buffer[0].violated and not violated:
                    self.violation_count -= 1
                elif not self.buffer[0].violated and violated:
                    self.violation_count += 1
                old_state_hash = self._hash_state(self.buffer[0].state)
                self.state_hashes.discard(old_state_hash)
                self.state_hashes.add(state_hash)
                heapq.heapreplace(self.buffer, example)
                return True
        return False

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
        n_samples = min(batch_size, len(self.buffer))
        weights = [math.exp(alpha * example.loss) for example in self.buffer]
        sampled_indices = random.choices(range(len(self.buffer)), weights=weights, k=n_samples)
        states = torch.stack([self.buffer[idx].state for idx in sampled_indices])
        losses = torch.tensor([self.buffer[idx].loss for idx in sampled_indices])
        
        return states, losses
    
    def __len__(self):
        """Return the current number of examples in the buffer."""
        return len(self.buffer)

def generate_random_states(n, mean=None, std=None, device='cpu'):
    """
    Generate random states with a specified distribution.
    
    Args:
        n: Number of states to generate
        mean: Mean tensor for state distribution (optional)
        std: Standard deviation tensor for state distribution (optional)
        device: Device to create tensors on
        
    Returns:
        Tensor of random states
    """
    if mean is None:
        mean = torch.tensor([0.0, 0.0], device=device)
    if std is None:
        std = torch.tensor([50.0, 15.0], device=device)  # CTE in cm, HE in degrees
        
    random_states = torch.randn(n, 2, device=device)
    random_states = random_states * std + mean
    
    return random_states

class RandomExampleSet(torch.utils.data.Dataset):
    """
    A dataset class that allows removal and addition of states during training.
    """
    def __init__(self, states, device='cpu'):
        """
        Initialize the dataset with a tensor of states.
        
        Args:
            states: Tensor of states with shape (n_samples, state_dim)
            device: Device to store tensors on
        """
        self.states = states.to(device)
        self.device = device
        
        # Calculate statistics for generating new random states
        self.mean = states.mean(dim=0).to(device)
        self.std = states.std(dim=0).to(device)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx]
    
    def remove_states(self, indices):
        """
        Remove states at the specified indices.
        
        Args:
            indices: List or tensor of indices to remove
            
        Returns:
            Tensor of removed states
        """
        if len(indices) == 0:
            return torch.zeros((0, self.states.size(1)), device=self.device)
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
        removed_states = self.states[indices].clone()
        keep_mask = torch.ones(len(self.states), dtype=torch.bool, device=self.device)
        keep_mask[indices] = False
        self.states = self.states[keep_mask]
        
        return removed_states
    
    def add_states(self, new_states):
        """
        Add new states to the dataset.
        
        Args:
            new_states: Tensor of states to add
        """
        if len(new_states) == 0:
            return
            
        self.states = torch.cat([self.states, new_states.to(self.device)], dim=0)
        
    def generate_and_add_states(self, n):
        """
        Generate and add random states with distribution similar to original dataset.
        
        Args:
            n: Number of states to generate and add
        """
        if n <= 0:
            return
            
        new_states = generate_random_states(n, self.mean, self.std, self.device)
        self.add_states(new_states)

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