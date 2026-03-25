"""
Experience Replay Buffer for DQN.
Stores transitions (state, action, reward, next_state, done) and samples batches.
"""

import numpy as np
from collections import deque
import random
from typing import Tuple


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    
    Experience replay breaks correlation between consecutive samples,
    which improves training stability.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state (4, 84, 84)
            action: Action taken
            reward: Reward received
            next_state: Next state (4, 84, 84)
            done: Whether episode ended
        """
        # Store as uint8 to save memory (75% reduction vs float32)
        # States are already 0-255 from preprocessing, so no data loss
        state_u8 = state.astype(np.uint8) if state.dtype != np.uint8 else state
        next_state_u8 = next_state.astype(np.uint8) if next_state.dtype != np.uint8 else next_state
        self.buffer.append((state_u8, action, reward, next_state_u8, done))
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Randomly sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert uint8 storage back to float32 for training
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


if __name__ == "__main__":
    # Quick test
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some dummy experiences
    for i in range(100):
        state = np.random.rand(4, 84, 84).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.rand(4, 84, 84).astype(np.float32)
        done = i % 20 == 0
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"Sampled states shape: {states.shape}")
    print(f"Sampled actions shape: {actions.shape}")
