"""
Convolutional Neural Network for DQN.
Takes 84x84x4 stacked frames and outputs Q-values for each action.
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network with CNN architecture.
    
    Input: (batch_size, 4, 84, 84) - 4 stacked grayscale frames
    Output: (batch_size, n_actions) - Q-values for each action
    """
    
    def __init__(self, n_actions: int = 3):
        """
        Initialize the DQN.
        
        Args:
            n_actions: Number of possible actions (3 for Pong: NOOP, UP, DOWN)
        """
        super(DQN, self).__init__()
        
        # Convolutional layers - extract features from frames
        self.conv = nn.Sequential(
            # Input: 4 x 84 x 84
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> 32 x 20 x 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> 64 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> 64 x 7 x 7
            nn.ReLU()
        )
        
        # Fully connected layers - compute Q-values
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # CRITICAL: Proper weight initialization (DeepMind standard)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using orthogonal initialization with proper gains."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Orthogonal init with sqrt(2) gain for ReLU
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Orthogonal init for linear layers
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4, 84, 84)
               Pixel values MUST be normalized to [0, 1] (done by environment)
        
        Returns:
            Q-values for each action, shape (batch_size, n_actions)
        """
        # NOTE: Normalization is now handled by PongEnvironment.get_stacked_state()
        # Input should already be in [0, 1] range
            
        # Extract features with CNN
        features = self.conv(x)
        
        # Flatten and compute Q-values
        features = features.view(features.size(0), -1)
        q_values = self.fc(features)
        
        return q_values


if __name__ == "__main__":
    # Quick test
    model = DQN(n_actions=3)
    dummy_input = torch.randn(1, 4, 84, 84)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Q-values: {output}")
