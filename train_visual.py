"""
Visual Training script for DQN Pong agent.
Shows the game window alongside a live-updating performance graph.
Automatically records training sessions for later review.

Uses multiprocessing to run the graph in a separate process,
avoiding conflicts between matplotlib and pygame.
"""

import os
import time
import numpy as np
import cv2
from collections import deque
from multiprocessing import Process, Queue
import signal
import sys

from agent import DQNAgent
from environment import PongEnvironment
from recorder import TrainingRecorder


def run_graph_process(data_queue: Queue, stop_queue: Queue):
    """
    Run the matplotlib graph in a separate process.
    This avoids conflicts with pygame's event loop.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
    # Data storage
    episodes = []
    rewards = []
    avg_rewards = []
    steps = []
    avg_steps = []
    
    # Setup the figure
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    fig.canvas.manager.set_window_title('Pong AI Training Progress - LIVE')
    
    # Configure reward plot
    ax_reward = axes[0]
    ax_reward.set_xlabel('Episode')
    ax_reward.set_ylabel('Reward')
    ax_reward.set_title('     Training Rewards', fontsize=11, fontweight='bold')
    ax_reward.grid(True, alpha=0.3)
    ax_reward.set_xlim(0, 50)
    ax_reward.set_ylim(-22, 22)
    
    line_reward, = ax_reward.plot([], [], 'cyan', alpha=0.6, linewidth=1.5, marker='o', markersize=3, label='Episode Reward')
    line_avg, = ax_reward.plot([], [], 'lime', linewidth=2.5, label='20-Episode Average')
    ax_reward.legend(loc='upper left')
    
    # Configure steps plot
    ax_steps = axes[1]
    ax_steps.set_xlabel('Episode')
    ax_steps.set_ylabel('Steps')
    ax_steps.set_title('Steps per Episode', fontsize=11, fontweight='bold')
    ax_steps.grid(True, alpha=0.3)
    ax_steps.set_xlim(0, 50)
    ax_steps.set_ylim(0, 3000)  # Auto-scaling will adjust this

    line_steps, = ax_steps.plot([], [], 'cyan', alpha=0.6, linewidth=1.5, marker='o', markersize=3, label='Steps')
    line_avg_steps, = ax_steps.plot([], [], 'lime', linewidth=2.5, label='20-Episode Average')
    ax_steps.legend(loc='upper left')
    
    # Add a status text
    status_text = fig.text(0.5, 0.02, 'Waiting for first episode...', ha='center', fontsize=10, color='yellow')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.ion()
    plt.show(block=False)
    
    last_update = time.time()
    
    try:
        while True:
            # Check for stop signal
            try:
                if not stop_queue.empty():
                    stop_queue.get_nowait()
                    break
            except:
                pass
            
            # Get new data from queue
            updated = False
            while not data_queue.empty():
                try:
                    data = data_queue.get_nowait()
                    if data is None:  # Stop signal
                        break
                    episodes.append(data['episode'])
                    rewards.append(data['reward'])
                    avg_rewards.append(data['avg_reward'])
                    steps.append(data['steps'])
                    avg_steps.append(data['avg_steps'])
                    updated = True
                except:
                    break
            
            # Update plot if we have new data
            if updated and episodes:
                line_reward.set_data(episodes, rewards)
                line_avg.set_data(episodes, avg_rewards)
                line_steps.set_data(episodes, steps)
                line_avg_steps.set_data(episodes, avg_steps)
                
                # Auto-scale x-axis
                max_ep = max(episodes)
                ax_reward.set_xlim(0, max(50, max_ep + 10))
                ax_steps.set_xlim(0, max(50, max_ep + 10))
                
                # Auto-scale y-axis for rewards
                if len(rewards) > 1:
                    min_r = min(rewards)
                    max_r = max(rewards)
                    margin = max(3, (max_r - min_r) * 0.2)
                    ax_reward.set_ylim(min_r - margin, max_r + margin)
                
                # Auto-scale y-axis for steps
                if len(steps) > 1:
                    max_s = max(steps)
                    ax_steps.set_ylim(0, max_s * 1.1)
                
                # Update status
                status_text.set_text(f'Episode {episodes[-1]} | Reward: {rewards[-1]:+.0f} | Avg: {avg_rewards[-1]:+.1f} | Steps: {steps[-1]}')
                status_text.set_color('lime' if avg_rewards[-1] > -15 else 'yellow')
                
                fig.canvas.draw_idle()
            
            # Keep window responsive
            fig.canvas.flush_events()
            plt.pause(0.1)
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Graph process error: {e}")
    finally:
        # Save final plot before closing
        if episodes:
            save_path = "checkpoints/training_progress.png"
            try:
                fig_save, axes_save = plt.subplots(2, 1, figsize=(12, 8))
                fig_save.patch.set_facecolor('#1a1a1a')
                
                axes_save[0].set_facecolor('#1a1a1a')
                axes_save[0].plot(episodes, rewards, 'cyan', alpha=0.6, linewidth=1, marker='o', markersize=2, label='Episode Reward')
                axes_save[0].plot(episodes, avg_rewards, 'lime', linewidth=2.5, label='20-Episode Average')
                axes_save[0].set_xlabel('Episode', color='white')
                axes_save[0].set_ylabel('Reward', color='white')
                axes_save[0].set_title('Training Rewards', fontsize=14, fontweight='bold', color='white')
                axes_save[0].legend(facecolor='#2a2a2a', edgecolor='gray')
                axes_save[0].grid(True, alpha=0.3)
                axes_save[0].tick_params(colors='white')
                
                axes_save[1].set_facecolor('#1a1a1a')
                axes_save[1].set_facecolor('#1a1a1a')
                axes_save[1].plot(episodes, steps, 'cyan', alpha=0.6, linewidth=1, marker='o', markersize=2, label='Steps')
                axes_save[1].plot(episodes, avg_steps, 'lime', linewidth=2.5, label='20-Episode Average')
                axes_save[1].set_xlabel('Episode', color='white')
                axes_save[1].set_ylabel('Steps', color='white')
                axes_save[1].set_title('Steps per Episode', fontsize=14, fontweight='bold', color='white')
                axes_save[1].legend(facecolor='#2a2a2a', edgecolor='gray')
                axes_save[1].grid(True, alpha=0.3)
                axes_save[1].tick_params(colors='white')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, facecolor='#1a1a1a')
                plt.close(fig_save)
                print(f"\nGraph saved to: {save_path}")
            except Exception as e:
                print(f"Could not save graph: {e}")
        
        plt.close('all')


def train_visual(
    num_episodes: int = 500,
    max_steps_per_episode: int = 10000,
    save_freq: int = 50,
    checkpoint_dir: str = "checkpoints",
    record: bool = False
):
    """
    Train the DQN agent with live visualization.
    Shows the Pong game window and a live-updating performance graph.
    """
    print("=" * 60)
    print("DQN Pong Training - VISUAL MODE")
    print("=" * 60)
    print("\nTwo windows will open:")
    print("  1. Pong game window - watch the AI learn to play")
    print("  2. Training graph - see performance improve over time")
    
    if record:
        print("[RECORDING] Video recording ENABLED")
    else:
        print("[NO RECORDING] Video recording DISABLED")

    print("\n" + "-" * 60)
    print("TO STOP: Press Ctrl+C in this terminal window")
    print("-" * 60)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create queues for communication with graph process
    data_queue = Queue()
    stop_queue = Queue()
    
    # Start the graph in a separate process
    print("\nStarting live graph window...")
    graph_process = Process(target=run_graph_process, args=(data_queue, stop_queue))
    graph_process.start()
    
    # Give the graph window time to initialize
    time.sleep(1.0)
    
    # Create environment with rgb_array mode so we can capture AND display frames
    # Using rgb_array instead of human allows us to record frames
    print("Starting Pong game window...")
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
    
    # ========================================
    # CRITICAL: Load existing checkpoint to continue training!
    # Without this, each run starts from scratch and loses all progress.
    # ========================================
    checkpoint_path = os.path.join(checkpoint_dir, "pong_dqn_final.pt")
    if os.path.exists(checkpoint_path):
        try:
            agent.load(checkpoint_path)
            print(f"[RESUMED] Loaded checkpoint from {checkpoint_path}")
            print(f"          Agent has {agent.steps_done} training steps")
            print(f"          Epsilon is at {agent.get_epsilon():.4f}")
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint: {e}")
            print("          Starting fresh training...")
    else:
        print("[NEW] No checkpoint found - starting fresh training")
    
    # Tracking
    recent_rewards = deque(maxlen=20)
    best_avg_reward = float('-inf')
    
    # Initialize recorder - ALWAYS load previous state for episode continuity
    recorder = TrainingRecorder(checkpoint_dir=checkpoint_dir, frame_skip=4)
    recorder.start()  # Load previous training history
    
    # Only capture frames if recording is enabled
    record_frames = record
    
    # Track starting episode number for continuous recording across sessions
    starting_episode = len(recorder.episodes) + 1
    
    start_time = time.time()
    
    print(f"\nTraining for {num_episodes} episodes...")
    print(f"Continuing from episode {starting_episode}")
    print(f"Device: {agent.device}")
    print(f"Initial epsilon: {agent.get_epsilon():.4f}")
    print(f"[WARMUP] Collecting {MIN_REPLAY_SIZE:,} samples before training starts...")
    print("\n" + "=" * 60)
    
    frames_captured = 0
    warmup_complete = False
    episode_lengths = []  # Track steps per episode for averaging
    
    # Create OpenCV window for displaying the game
    cv2.namedWindow("Pong AI Training", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pong AI Training", 640, 480)
    
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
                
                # Get frame from environment for display AND recording
                try:
                    raw_frame = env.env.render()
                    if isinstance(raw_frame, np.ndarray) and raw_frame.size > 0:
                        # Display the frame using OpenCV
                        # Convert RGB to BGR for OpenCV
                        display_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Pong AI Training", display_frame)
                        cv2.waitKey(1)  # Process window events (1ms delay)
                        
                        # Record frame if recording is enabled
                        if record:
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
            recent_rewards.append(episode_reward)
            episode_lengths.append(step + 1)  # Track steps for average
            avg_reward = np.mean(recent_rewards)
            avg_steps = np.mean(episode_lengths[-20:]) if episode_lengths else 0  # 20-episode avg
            
            # Update recorder graph
            recorder.update_graph(
                episode=episode,
                reward=episode_reward,
                avg_reward=avg_reward,
                epsilon=agent.get_epsilon(),
                steps=step + 1
            )
            
            # Send data to graph process
            data_queue.put({
                'episode': episode,
                'reward': episode_reward,
                'avg_reward': avg_reward,
                'epsilon': agent.get_epsilon(),
                'steps': step + 1,
                'avg_steps': avg_steps
            })
            
            # Console output
            elapsed = time.time() - start_time
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:+6.0f} | "
                  f"Avg(20): {avg_reward:+6.1f} | "
                  f"Epsilon: {agent.get_epsilon():.3f} | "
                  f"Steps: {step+1:4d} | "
                  f"Steps Avg(20): {avg_steps:5.0f} | "
                  f"Time: {elapsed/60:.1f}m")
            
            # Print Q-value diagnostics every 20 episodes to monitor learning
            # Print Q-value diagnostics every 20 episodes to monitor learning
            if False and episode % 20 == 0:  # Diagnostics disabled for cleaner output
                q_debug = agent.get_q_values_debug(state)
                print(f"  [Q-DEBUG] NOOP:{q_debug['q_noop']:7.2f} | "
                      f"UP:{q_debug['q_up']:7.2f} | "
                      f"DOWN:{q_debug['q_down']:7.2f} | "
                      f"Spread:{q_debug['q_spread']:.4f}")
                
                # Buffer diagnostics - verify memory is healthy
                print(f"  [BUFFER] Size: {len(agent.buffer):,}/{agent.buffer.buffer.maxlen:,} | "
                      f"Training steps: {agent.steps_done:,}")
                
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
            
            # Save checkpoint
            if episode % save_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"pong_dqn_ep{episode}.pt")
                agent.save(checkpoint_path)
                print(f"  --> Checkpoint saved: {checkpoint_path}")
                
            # Save best model
            if avg_reward > best_avg_reward and len(recent_rewards) >= 20:
                best_avg_reward = avg_reward
                best_path = os.path.join(checkpoint_dir, "pong_dqn_best.pt")
                agent.save(best_path)
                print(f"  *** New best average reward: {best_avg_reward:.1f} ***")
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Training stopped by user (Ctrl+C)")
        print("=" * 60)
    
    finally:
        # Save final checkpoint
        final_path = os.path.join(checkpoint_dir, "pong_dqn_final.pt")
        agent.save(final_path)
        
        # Save recorder state (preserves frames and graph data)
        recorder.save()
        
        # Only show video export message if frames were captured
        if record and frames_captured > 0:
            print(f"\n[VIDEO] {frames_captured:,} frames captured. Export with: python main.py export-video")
        
        recorder.close()
        
        # Close environment and OpenCV window
        env.close()
        cv2.destroyAllWindows()
        
        # Stop graph process gracefully
        print("\nStopping graph window...")
        stop_queue.put(True)
        graph_process.join(timeout=3)
        if graph_process.is_alive():
            graph_process.terminate()
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Episodes completed: {episode if 'episode' in dir() else 0}")
        print(f"  Frames captured: {frames_captured}")
        print(f"  Total time: {total_time/60:.1f} minutes")
        if recent_rewards:
            print(f"  Final avg reward (last 20): {np.mean(list(recent_rewards)):.1f}")
        print(f"  Best avg reward: {best_avg_reward:.1f}")
        print(f"  Model saved to: {final_path}")
        print(f"  Graph saved to: {checkpoint_dir}/training_progress.png")
        print("=" * 60)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    
    train_visual(
        num_episodes=500,
        max_steps_per_episode=10000,
        save_freq=50,
        checkpoint_dir="checkpoints"
    )
