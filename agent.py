"""
DQN Agent that learns to play Pong.
Implements epsilon-greedy action selection and the DQN training algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional
import os

from model import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Learning agent for Atari games.
    
    Uses two networks:
    - policy_net: The main network being trained
    - target_net: A copy updated periodically for stable target Q-values
    """
    
    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 100000,
        buffer_capacity: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        device: str = "cpu"
    ):
        """
        Initialize the DQN agent.
        
        Args:
            n_actions: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps over which epsilon decays
            buffer_capacity: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: How often to update target network (in steps)
            device: Device to use ('cpu' or 'cuda')
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        
        # Networks
        self.policy_net = DQN(n_actions).to(self.device)
        self.target_net = DQN(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is not trained directly
        
        # Optimizer and loss - RMSprop with DeepMind's exact hyperparameters
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(), 
            lr=learning_rate,
            alpha=0.95,      # Smoothing constant (DeepMind used 0.95)
            eps=0.01         # Epsilon for numerical stability (DeepMind used 0.01)
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)
        
        # Tracking
        self.steps_done = 0
        
    def get_epsilon(self) -> float:
        """Calculate current epsilon using LINEAR decay (standard for DQN)."""
        # Linear decay from epsilon_start to epsilon_end over epsilon_decay steps
        progress = min(1.0, self.steps_done / self.epsilon_decay)
        epsilon = self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)
        return max(self.epsilon_end, epsilon)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state (4, 84, 84)
            training: Whether to use epsilon-greedy (True) or greedy (False)
            
        Returns:
            Selected action index
        """
        epsilon = self.get_epsilon() if training else 0.0
        
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action according to policy
            with torch.no_grad():
                # Normalize uint8 [0,255] to float [0,1] here!
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using a batch from the replay buffer.
        
        Returns:
            Loss value if training occurred, None if buffer too small
        """
        if len(self.buffer) < self.batch_size:
            return None
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors and NORMALIZE states from uint8 [0,255] to float [0,1]
        states = torch.FloatTensor(states).to(self.device) / 255.0
        actions = torch.LongTensor(actions).to(self.device)
        # CRITICAL: Squeeze to ensure [batch_size] shape, fixes broadcasting bug!
        rewards = torch.FloatTensor(rewards).to(self.device).squeeze()
        next_states = torch.FloatTensor(next_states).to(self.device) / 255.0
        dones = torch.FloatTensor(dones).to(self.device).squeeze()
        
        # Current Q-values: Q(s, a)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use policy net to SELECT best action, target net to EVALUATE it
        # This reduces overestimation bias compared to vanilla DQN
        with torch.no_grad():
            # Policy network selects the best action for next states
            next_actions = self.policy_net(next_states).argmax(dim=1)
            # Target network evaluates the Q-value of that action
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Bellman target: r + gamma * Q_target(s', argmax_a Q_policy(s', a))
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability (1.0 is standard for Huber loss)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        
        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
    
    def get_q_values_debug(self, state: np.ndarray) -> dict:
        """
        Get Q-values for debugging purposes.
        Returns dict with Q-values for each action and other diagnostics.
        """
        with torch.no_grad():
            # Normalize uint8 [0,255] to float [0,1]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
            q_values = self.policy_net(state_tensor).squeeze(0)
            
            return {
                'q_noop': q_values[0].item(),
                'q_up': q_values[1].item(),
                'q_down': q_values[2].item(),
                'q_max': q_values.max().item(),
                'q_min': q_values.min().item(),
                'q_spread': (q_values.max() - q_values.min()).item(),
                'best_action': q_values.argmax().item()
            }
    
    # ========================================
    # ADVANCED DIAGNOSTICS
    # ========================================
    
    def get_buffer_stats(self) -> dict:
        """Analyze replay buffer contents."""
        if len(self.buffer) == 0:
            return {'empty': True}
        
        # Sample rewards from buffer to check distribution
        rewards = []
        for i in range(min(1000, len(self.buffer))):
            idx = np.random.randint(len(self.buffer))
            _, _, r, _, _ = self.buffer.buffer[idx]
            rewards.append(r)
        
        rewards = np.array(rewards)
        neg_pct = (rewards < 0).sum() / len(rewards) * 100
        zero_pct = (rewards == 0).sum() / len(rewards) * 100
        pos_pct = (rewards > 0).sum() / len(rewards) * 100
        
        return {
            'size': len(self.buffer),
            'neg_reward_pct': neg_pct,
            'zero_reward_pct': zero_pct,
            'pos_reward_pct': pos_pct
        }
    
    def get_gradient_stats(self) -> dict:
        """Check gradient magnitudes after a training step."""
        grad_stats = {}
        for name, param in self.policy_net.named_parameters():
            if param.grad is not None:
                grad_stats[name] = param.grad.abs().mean().item()
        return grad_stats
    
    def get_checkpoint_hash(self) -> str:
        """Get a hash of current weights for verification."""
        import hashlib
        weight_sum = 0.0
        param_count = 0
        for param in self.policy_net.parameters():
            weight_sum += param.data.sum().item()
            param_count += param.numel()
        # Create a simple hash from weight statistics
        hash_str = f"{weight_sum:.6f}_{param_count}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:12]
    
    def get_target_network_diff(self) -> float:
        """Calculate difference between policy and target networks."""
        total_diff = 0.0
        param_count = 0
        for (p_param, t_param) in zip(self.policy_net.parameters(), 
                                       self.target_net.parameters()):
            total_diff += (p_param.data - t_param.data).abs().sum().item()
            param_count += p_param.numel()
        return total_diff / param_count if param_count > 0 else 0.0
    
    def get_next_target_sync(self) -> dict:
        """Get info about target network synchronization."""
        last_sync = (self.steps_done // self.target_update_freq) * self.target_update_freq
        next_sync = last_sync + self.target_update_freq
        return {
            'last_sync': last_sync,
            'next_sync': next_sync,
            'steps_until_sync': next_sync - self.steps_done,
            'policy_target_diff': self.get_target_network_diff()
        }
    
    def train_step_with_diagnostics(self) -> dict:
        """Training step that returns before/after Q-value comparison."""
        if len(self.buffer) < self.batch_size:
            return {'trained': False}
        
        # Sample a fixed state for before/after comparison
        sample_state, _, _, _, _ = self.buffer.buffer[0]
        
        # Get Q-values BEFORE training
        with torch.no_grad():
            state_tensor = torch.FloatTensor(sample_state).unsqueeze(0).to(self.device) / 255.0
            q_before = self.policy_net(state_tensor).squeeze().clone()
        
        # Do actual training step
        loss = self.train_step()
        
        # Get Q-values AFTER training
        with torch.no_grad():
            q_after = self.policy_net(state_tensor).squeeze()
        
        # Calculate change
        q_change = (q_after - q_before).abs().mean().item()
        
        # Get gradient stats (must be called right after backward)
        grad_stats = self.get_gradient_stats()
        avg_grad = np.mean(list(grad_stats.values())) if grad_stats else 0.0
        
        return {
            'trained': True,
            'loss': loss,
            'q_change': q_change,
            'avg_gradient': avg_grad
        }
    
    def save(self, path: str):
        """Save the agent's state."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        weight_hash = self.get_checkpoint_hash()
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'weight_hash': weight_hash,  # For verification
        }, path)
        print(f"Agent saved to {path}")
        
    def load(self, path: str) -> dict:
        """Load the agent's state and return verification info."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        
        # Verify checkpoint
        saved_hash = checkpoint.get('weight_hash', 'unknown')
        loaded_hash = self.get_checkpoint_hash()
        hash_matches = saved_hash == loaded_hash
        
        print(f"Agent loaded from {path}")
        
        return {
            'saved_hash': saved_hash,
            'loaded_hash': loaded_hash,
            'hash_matches': hash_matches,
            'param_count': sum(p.numel() for p in self.policy_net.parameters())
        }


if __name__ == "__main__":
    # Quick test
    agent = DQNAgent(n_actions=3)
    
    # Simulate some transitions
    for i in range(100):
        state = np.random.rand(4, 84, 84).astype(np.float32)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.rand(4, 84, 84).astype(np.float32)
        done = i % 20 == 0
        
        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train_step()
        
    print(f"Steps done: {agent.steps_done}")
    print(f"Current epsilon: {agent.get_epsilon():.4f}")
    print(f"Buffer size: {len(agent.buffer)}")
