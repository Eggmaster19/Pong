"""
Play script - watch the trained agent play Pong!
"""

import argparse
import time
import numpy as np

from agent import DQNAgent
from environment import PongEnvironment

# Evaluation epsilon - mostly exploit, tiny exploration for edge cases
EVAL_EPSILON = 0.0


def play(
    checkpoint_path: str = "checkpoints/pong_dqn_best.pt",
    num_episodes: int = 5,
    delay: float = 0.01
):
    """
    Watch the trained agent play Pong.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        num_episodes: Number of episodes to play
        delay: Delay between frames (seconds)
    """
    print("=" * 60)
    print("DQN Pong - Watch Mode")
    print("=" * 60)
    
    # Create environment with rendering
    env = PongEnvironment(render_mode="human")
    
    # Create and load agent
    agent = DQNAgent(n_actions=env.n_actions, device="cpu")
    
    try:
        agent.load(checkpoint_path)
        print(f"Loaded model from: {checkpoint_path}")
    except FileNotFoundError:
        print(f"No checkpoint found at {checkpoint_path}")
        print("Playing with random agent instead...")
    
    # Set evaluation epsilon for near-optimal play
    agent.epsilon_start = EVAL_EPSILON
    agent.epsilon_end = EVAL_EPSILON
    agent.steps_done = 1000000  # Force epsilon to end value
    print(f"Evaluation epsilon: {EVAL_EPSILON}")
    
    print(f"\nPlaying {num_episodes} episodes...")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    episode_rewards = []
    
    try:
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                # Select action with eval epsilon (mostly exploit)
                action = agent.select_action(state, training=True)  # Use epsilon
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                # Small delay for visualization
                if delay > 0:
                    time.sleep(delay)
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            result = "WIN" if episode_reward > 0 else "LOSS" if episode_reward < 0 else "DRAW"
            print(f"Episode {episode}: Reward = {episode_reward:+.0f} ({result}) | Steps: {steps}")
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    env.close()
    
    if episode_rewards:
        print("\n" + "=" * 60)
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print(f"Win rate: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards) * 100:.1f}%")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained DQN agent play Pong")
    parser.add_argument("--checkpoint", "-c", type=str, 
                        default="checkpoints/pong_dqn_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", "-e", type=int, default=5,
                        help="Number of episodes to play")
    parser.add_argument("--delay", "-d", type=float, default=0.01,
                        help="Delay between frames (seconds)")
    
    args = parser.parse_args()
    play(args.checkpoint, args.episodes, args.delay)

