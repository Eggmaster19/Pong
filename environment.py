"""
Environment wrapper for Atari Pong.
Handles frame preprocessing, stacking, and action mapping.

Implements critical fixes for reliable learning:
- Max-frame merging to handle sprite flickering
- Cropping to remove scoreboard and focus on play area
- Proper frame stacking for temporal information
"""

import gymnasium as gym
import ale_py  # Register ALE environments with gymnasium
import numpy as np
import cv2
from collections import deque
from typing import Tuple, Optional


class PongEnvironment:
    """
    Wrapper for the Pong Atari environment.
    
    Preprocessing:
    - Max-merges last 2 raw frames to handle sprite flickering
    - Crops to play area (removes scoreboard)
    - Converts to grayscale
    - Resizes to 84x84
    - Stacks 4 consecutive frames
    - Maps to simplified action space (NOOP, UP, DOWN)
    """
    
    # Pong action mapping: reduce from 6 actions to 3
    # Original: 0=NOOP, 1=FIRE, 2=UP, 3=DOWN, 4=UPFIRE, 5=DOWNFIRE
    # We map to: 0=NOOP, 1=UP, 2=DOWN
    ACTION_MAP = {0: 0, 1: 2, 2: 3}
    
    # Crop region for Pong (removes scoreboard at top)
    # Original frame is 210x160, we crop to the play area
    CROP_TOP = 34
    CROP_BOTTOM = 194
    
    def __init__(self, render_mode: Optional[str] = None, frame_stack: int = 4):
        """
        Initialize the Pong environment.
        
        Args:
            render_mode: 'human' for visualization, 'rgb_array' for frame capture, None for training
            frame_stack: Number of frames to stack (default 4)
        """
        # IMPORTANT: frameskip=1 so we can manually handle frame merging
        self.env = gym.make(
            "ALE/Pong-v5",
            render_mode=render_mode,
            frameskip=1,  # We handle frameskip manually for proper max-merging
            repeat_action_probability=0.0  # Deterministic
        )
        
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.n_actions = 3  # NOOP, UP, DOWN
        
        # Store last 2 raw frames for max-merging (handles flickering)
        self.raw_frames = deque(maxlen=2)
        
        # Number of frames to skip (repeat action)
        self.frameskip = 4
        
    def _max_merge_frames(self) -> np.ndarray:
        """
        Take pixel-wise maximum of last 2 frames.
        This handles sprite flickering where objects appear on alternate frames.
        """
        if len(self.raw_frames) == 1:
            return self.raw_frames[0]
        return np.maximum(self.raw_frames[0], self.raw_frames[1])
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame.
        
        - Crop to play area (removes scoreboard)
        - Convert to grayscale
        - Resize to 84x84
        - Return as uint8 (0-255)
        """
        # Crop to play area (removes scoreboard at top, floor at bottom)
        cropped = frame[self.CROP_TOP:self.CROP_BOTTOM, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        return resized.astype(np.uint8)
    
    def get_stacked_state(self) -> np.ndarray:
        """Return the stacked frames as a (4, 84, 84) uint8 array.
        
        NOTE: Returns uint8 to save RAM in replay buffer (4x less memory).
        The agent will normalize to [0,1] when feeding to the network.
        """
        stacked = np.array(self.frames, dtype=np.uint8)
        return stacked
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return initial state.
        
        Returns:
            Stacked frames of shape (4, 84, 84), as uint8
        """
        frame, _ = self.env.reset()
        
        # CRITICAL: Force FIRE action to start the game immediately!
        # Without this, the game waits for the opponent to serve (huge delay)
        frame, _, _, _, _ = self.env.step(1)  # Action 1 = FIRE
        
        # Clear raw frame buffer and add initial frame
        self.raw_frames.clear()
        self.raw_frames.append(frame)
        
        # Process the initial frame
        processed = self.preprocess_frame(frame)
        
        # Initialize frame stack with the same frame
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(processed)
            
        return self.get_stacked_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take an action in the environment with frameskip and max-merging.
        
        Args:
            action: Action index (0=NOOP, 1=UP, 2=DOWN)
            
        Returns:
            (next_state, reward, terminated, truncated, info)
        """
        # Map to actual Atari action
        atari_action = self.ACTION_MAP[action]
        
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Repeat action for frameskip frames, storing last 2 for max-merging
        for i in range(self.frameskip):
            frame, reward, terminated, truncated, info = self.env.step(atari_action)
            total_reward += reward
            
            # Store last 2 frames for max-merging
            if i >= self.frameskip - 2:
                self.raw_frames.append(frame)
            
            if terminated or truncated:
                break
        
        # Max-merge the last 2 frames to handle flickering
        merged_frame = self._max_merge_frames()
        
        # Preprocess and add to frame stack
        processed = self.preprocess_frame(merged_frame)
        self.frames.append(processed)
        
        return self.get_stacked_state(), total_reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Get the current raw frame for rendering/recording."""
        if len(self.raw_frames) > 0:
            return self._max_merge_frames()
        return None
    
    def close(self):
        """Close the environment."""
        self.env.close()


def test_environment():
    """Test the environment wrapper."""
    env = PongEnvironment(render_mode=None)
    
    print(f"Number of actions: {env.n_actions}")
    
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"State min: {state.min():.3f}, max: {state.max():.3f}")  # Should be 0-1
    
    # Run a few steps
    total_reward = 0
    for i in range(100):
        action = np.random.randint(env.n_actions)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            break
            
    print(f"Total reward: {total_reward}")
    print(f"Final state shape: {next_state.shape}")
    print(f"State min: {next_state.min():.3f}, max: {next_state.max():.3f}")
    
    env.close()
    print("Environment test passed!")


if __name__ == "__main__":
    test_environment()
