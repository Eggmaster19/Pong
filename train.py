"""
Training script for DQN Pong agent.
Runs the main training loop with progress tracking.
Automatically records training sessions for later review.
"""

import os
import time
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import cv2
import psutil

from agent import DQNAgent
from environment import PongEnvironment
from recorder import TrainingRecorder


def train(
    num_episodes: int = 1000,
    max_steps_per_episode: int = 10000,
    save_freq: int = 50,
    checkpoint_dir: str = "checkpoints",
    log_freq: int = 10,
    record: bool = False
):
    """
    Train the DQN agent to play Pong.
    
    Args:
        num_episodes: Total number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        save_freq: Save checkpoint every N episodes
        checkpoint_dir: Directory to save checkpoints
        log_freq: Print progress every N episodes
        record: Whether to record video of the training session
    """
    print("=" * 60)
    print("DQN Pong Training")
    print("=" * 60)
    
    if record:
        print("[RECORDING] Video recording ENABLED - Training will be slower")
    else:
        print("[NO RECORDING] Video recording DISABLED - Training will be faster")
    
    # Create environment with rgb_array mode so we can capture frames for recording
    env = PongEnvironment(render_mode="rgb_array")
    agent = DQNAgent(
        n_actions=env.n_actions,
        learning_rate=0.00025,       # DeepMind standard for RMSprop
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,             # DeepMind standard (was 0.02)
        epsilon_decay=250000,        # Slower decay for better exploration
        buffer_capacity=1000000,     # DeepMind standard (1M transitions)
        batch_size=32,
        target_update_freq=10000,    # DeepMind standard
        device="cpu"
    )
    
    # CRITICAL: Warmup period - don't train until buffer has diverse samples
    # DeepMind used 50000 frames before training starts
    MIN_REPLAY_SIZE = 50000
    TRAIN_FREQUENCY = 4  # Train every 4 steps (DeepMind standard)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ========================================
    # CRITICAL: Load existing checkpoint to continue training!
    # Without this, each run starts from scratch and loses all progress.
    # ========================================
    checkpoint_path = os.path.join(checkpoint_dir, "pong_dqn_final.pt")
    if os.path.exists(checkpoint_path):
        try:
            load_info = agent.load(checkpoint_path)
            print(f"[RESUMED] Loaded checkpoint from {checkpoint_path}")
            print(f"          Agent has {agent.steps_done} training steps")
            print(f"          Epsilon is at {agent.get_epsilon():.4f}")
            print(f"          [CHECKPOINT] Params: {load_info['param_count']:,} | "
                  f"Hash: {load_info['loaded_hash']} | "
                  f"Verified: {'✓' if load_info['hash_matches'] else '✗'}")
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint: {e}")
            print("          Starting fresh training...")
    else:
        print("[NEW] No checkpoint found - starting fresh training")
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    recent_rewards = deque(maxlen=20)  # Use 20-episode window for average
    epsilons = []  # Track epsilon values for graphing
    best_avg_reward = float('-inf')
    
    # Initialize recorder - ALWAYS load previous state for episode continuity
    recorder = TrainingRecorder(checkpoint_dir=checkpoint_dir, frame_skip=8)
    recorder.start()  # Load previous training history
    
    # Only capture frames if recording is enabled
    record_frames = record
    
    # Track starting episode number for continuous recording across sessions
    starting_episode = len(recorder.episodes) + 1
    
    start_time = time.time()
    
    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Continuing from episode {starting_episode}")
    print(f"Device: {agent.device}")
    print(f"Initial epsilon: {agent.get_epsilon():.4f}")
    # Prevent duplicate warmup messages if train() is somehow called twice
    if os.environ.get("PONG_WARMUP_SHOWN") != "1":
        print(f"[WARMUP] Collecting {MIN_REPLAY_SIZE:,} samples before training starts...")
        os.environ["PONG_WARMUP_SHOWN"] = "1"
    print("Press Ctrl+C at any time to stop training and save progress.")
    print("-" * 60)
    
    frames_captured = 0
    warmup_complete = False
    
    try:
        for episode_idx in range(1, num_episodes + 1):
            # Use continuous episode numbering
            episode = starting_episode + episode_idx - 1
            
            state = env.reset()
            episode_reward = 0
            episode_loss = []
            action_counts = [0, 0, 0]  # Track action distribution: [NOOP, UP, DOWN]
            
            for step in range(max_steps_per_episode):
                # Select action
                action = agent.select_action(state, training=True)
                
                # Track action distribution for diagnostics
                action_counts[action] += 1
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Reward clipping to [-1, 1] for stability (DeepMind standard)
                reward = max(-1.0, min(reward, 1.0))
                
                # Capture RGB frame for recording
                if record:
                    try:
                        raw_frame = env.env.render()
                        if raw_frame is not None and isinstance(raw_frame, np.ndarray) and raw_frame.size > 0:
                            recorder.add_frame(raw_frame)
                            frames_captured += 1
                    except Exception as e:
                        pass  # Continue even if frame capture fails
                
                # EPISODIC LIVES: Mark any point scored as terminal for LEARNING
                # This is from DeepMind paper: "for games where there is a life counter...
                # used to mark the end of an episode during training"
                # Key insight: The TRANSITION is marked terminal, but the game continues
                life_lost = (reward != 0)  # Any point scored = "life" for Pong
                agent.store_transition(state, action, reward, next_state, done or life_lost)
                
                # Only train after warmup AND every TRAIN_FREQUENCY steps (DeepMind standard)
                if len(agent.buffer) >= MIN_REPLAY_SIZE and step % TRAIN_FREQUENCY == 0:
                    if not warmup_complete:
                        print(f"[WARMUP COMPLETE] Buffer has {len(agent.buffer):,} samples - training started!")
                        warmup_complete = True
                    loss = agent.train_step()
                    if loss is not None:
                        episode_loss.append(loss)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Track progress
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            recent_rewards.append(episode_reward)
            epsilons.append(agent.get_epsilon())  # Track epsilon for graph
            
            avg_reward = np.mean(recent_rewards)  # 20-episode average
            avg_steps = np.mean(episode_lengths[-20:]) if episode_lengths else 0  # 20-episode avg steps
            
            # Update recorder graph (always update graph data regardless of recording frames)
            recorder.update_graph(
                episode=episode,
                reward=episode_reward,
                avg_reward=avg_reward,
                epsilon=agent.get_epsilon(),
                steps=step + 1  # Pass steps explicitly
            )
            
            # Log progress every 5 episodes
            if episode_idx % 5 == 0:
                elapsed = time.time() - start_time
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:6.1f} | "
                      f"Avg(20): {avg_reward:6.2f} | "
                      f"Steps Avg(20): {avg_steps:5.0f} | "
                      f"Time: {elapsed/60:.1f}m")
            
            # Memory monitoring every 100 episodes to detect memory issues
            if episode_idx % 100 == 0:
                mem = psutil.Process().memory_info()
                print(f"  [MEMORY] Process: {mem.rss / 1024**3:.2f} GB | "
                      f"Buffer: {len(agent.buffer):,}/{agent.buffer.buffer.maxlen:,}")
            
            # Print Q-value diagnostics every 20 episodes to monitor learning
            # Print Q-value diagnostics every 20 episodes to monitor learning
            if False and episode_idx % 20 == 0:  # Diagnostics disabled for cleaner output
                q_debug = agent.get_q_values_debug(state)
                print(f"  [Q-DEBUG] NOOP:{q_debug['q_noop']:7.2f} | "
                      f"UP:{q_debug['q_up']:7.2f} | "
                      f"DOWN:{q_debug['q_down']:7.2f} | "
                      f"Spread:{q_debug['q_spread']:.4f}")
                
                # Buffer diagnostics - verify memory is healthy
                print(f"  [BUFFER] Size: {len(agent.buffer):,}/{agent.buffer.buffer.maxlen:,} | "
                      f"Training steps: {agent.steps_done:,}")
                
                # NEW: Buffer content analysis - what rewards are stored
                buffer_stats = agent.get_buffer_stats()
                if not buffer_stats.get('empty'):
                    print(f"  [BUFFER-CONTENT] Rewards: -1:{buffer_stats['neg_reward_pct']:.1f}% | "
                          f"0:{buffer_stats['zero_reward_pct']:.1f}% | "
                          f"+1:{buffer_stats['pos_reward_pct']:.1f}%")
                
                # Loss diagnostics - verify gradients are flowing
                if episode_loss:
                    avg_loss = np.mean(episode_loss[-100:]) if len(episode_loss) > 100 else np.mean(episode_loss)
                    print(f"  [LOSS] Avg(last 100): {avg_loss:.6f}")
                
                # Action distribution - detect if agent is stuck on one action
                total_actions = sum(action_counts) if sum(action_counts) > 0 else 1
                print(f"  [ACTIONS] NOOP:{action_counts[0]/total_actions*100:5.1f}% | "
                      f"UP:{action_counts[1]/total_actions*100:5.1f}% | "
                      f"DOWN:{action_counts[2]/total_actions*100:5.1f}%")
                
                # Weight magnitude check - detect dead neurons or exploding weights
                policy_params = list(agent.policy_net.parameters())
                weight_norms = [p.data.norm().item() for p in policy_params]
                print(f"  [WEIGHTS] Min norm:{min(weight_norms):.2f} | "
                      f"Max norm:{max(weight_norms):.2f} | "
                      f"Layers:{len(weight_norms)}")
                
                # NEW: Target network sync info
                sync_info = agent.get_next_target_sync()
                print(f"  [TARGET-NET] Last sync:{sync_info['last_sync']:,} | "
                      f"Next:{sync_info['next_sync']:,} | "
                      f"Policy-Target diff:{sync_info['policy_target_diff']:.6f}")
            
            # Save checkpoint
            if episode_idx % save_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"pong_dqn_ep{episode}.pt")
                agent.save(checkpoint_path)
                
            # Save best model
            if avg_reward > best_avg_reward and len(recent_rewards) >= 100:
                best_avg_reward = avg_reward
                best_path = os.path.join(checkpoint_dir, "pong_dqn_best.pt")
                agent.save(best_path)
                print(f"  *** New best average reward: {best_avg_reward:.2f} ***")
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Training interrupted by user!")
        print("=" * 60)
    
    finally:
        # Always save progress, even on interrupt
        final_path = os.path.join(checkpoint_dir, "pong_dqn_final.pt")
        agent.save(final_path)
        
        # Save recorder state (preserves frames and graph data)
        recorder.save()
        
        # Only show video export message if frames were captured
        if record and frames_captured > 0:
            print(f"\n[VIDEO] {frames_captured:,} frames captured. Export with: python main.py export-video")
        
        # Plot training progress using RECORDER data (includes ALL episodes, not just this session)
        # This fixes the graph glitch where only current session data was shown
        if recorder.rewards:
            # Update to pass lengths instead of epsilons
            save_training_plots(recorder.rewards, recorder.episode_lengths, checkpoint_dir)
        elif episode_rewards:
            save_training_plots(episode_rewards, epsilons, checkpoint_dir)
                
        recorder.close()
        env.close()
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Summary")
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Frames captured for video: {frames_captured}")
        print(f"Total time: {total_time/60:.1f} minutes")
        if recent_rewards:
            print(f"Final avg reward (last 20): {np.mean(list(recent_rewards)):.2f}")
        print(f"Model saved to: {final_path}")
        print(f"Graph saved to: {checkpoint_dir}/training_progress.png")
        print("=" * 60)
    
    return episode_rewards, epsilons


def save_training_plots(rewards, lengths, save_dir):
    """Save training progress plots with dark theme matching train-visual."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.patch.set_facecolor('#1a1a1a')
    
    # Top plot: Rewards
    axes[0].set_facecolor('#1a1a1a')
    axes[0].plot(rewards, 'cyan', alpha=0.5, linewidth=1, label='Episode Reward')
    
    # 20-episode rolling average
    window = 20
    if len(rewards) >= window:
        rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), rolling_avg, 
                     'lime', linewidth=2.5, label='20-Episode Average')
    
    axes[0].set_xlabel('Episode', color='white')
    axes[0].set_ylabel('Reward', color='white')
    axes[0].set_title('             Training Rewards', fontsize=14, fontweight='bold', color='white')
    legend = axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, 1.15), 
                           ncol=2, facecolor='#2a2a2a', edgecolor='gray')
    plt.setp(legend.get_texts(), color='white')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(colors='white')
    
    # Bottom plot: Steps (Episode Length)
    axes[1].set_facecolor('#1a1a1a')
    if lengths:
        axes[1].plot(lengths, 'orange', alpha=0.5, linewidth=1, label='Steps')
        
        # 20-episode rolling average
        if len(lengths) >= window:
            rolling_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(lengths)), rolling_len,
                         'yellow', linewidth=2.5, label='20-Episode Average')
            
    axes[1].set_xlabel('Episode', color='white')
    axes[1].set_ylabel('Steps', color='white')
    axes[1].set_title('Episode Length', fontsize=14, fontweight='bold', color='white')
    legend = axes[1].legend(loc='upper left', bbox_to_anchor=(0.0, 1.15),
                           ncol=2, facecolor='#2a2a2a', edgecolor='gray')
    plt.setp(legend.get_texts(), color='white')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=150, facecolor='#1a1a1a')
    plt.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DQN Pong agent")
    parser.add_argument("--episodes", "-e", type=int, default=500,
                        help="Number of training episodes (default: 500)")
    parser.add_argument("--checkpoint-dir", "-d", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--record", "-r", action="store_true",
                        help="Enable video recording")
    args = parser.parse_args()
    
    train(
        num_episodes=args.episodes,
        max_steps_per_episode=10000,
        save_freq=50,
        checkpoint_dir=args.checkpoint_dir,
        log_freq=10,
        record=args.record
    )
