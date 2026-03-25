"""
Evaluation modes for the trained DQN Pong agent.

Modes:
- watch: Visual playback with game window and live graph
- versus: Human vs AI with keyboard controls
- benchmark: Background evaluation with terminal output and final graphs

All modes use epsilon=0.01 for near-optimal exploitation with slight exploration.
"""

import os
import time
import numpy as np
import cv2
from collections import deque

from agent import DQNAgent
from environment import PongEnvironment
from recorder import TrainingRecorder


# Evaluation epsilon - 0.0, DeepMind standard is 0.05
# Ensures agent doesn't get stuck in loops but still mostly exploits
EVAL_EPSILON = 0.0


def watch_agent(
    checkpoint_path: str = "checkpoints/pong_dqn_final.pt",
    num_episodes: int = 10,
    checkpoint_dir: str = "checkpoints"
):
    """
    Watch the trained agent play with visual game window and live performance graph.
    Like train_visual but without training - pure evaluation.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from multiprocessing import Process, Queue
    
    print("=" * 60)
    print("DQN Pong - Watch Mode (Visual)")
    print("=" * 60)
    print(f"Evaluation epsilon: {EVAL_EPSILON}")
    print(f"Evaluation conditions: 0-30 random no-ops at start")
    
    # Create environment with RGB output for display
    env = PongEnvironment(render_mode="rgb_array")
    
    # Create and load agent with custom epsilon
    agent = DQNAgent(n_actions=env.n_actions, device="cpu")
    
    try:
        agent.load(checkpoint_path)

        print(f"Agent has {agent.steps_done:,} training steps")
    except FileNotFoundError:
        print(f"No checkpoint found at {checkpoint_path}")
        print("Using random agent...")
    
    # Force evaluation epsilon
    original_epsilon_end = agent.epsilon_end
    agent.epsilon_start = EVAL_EPSILON
    agent.epsilon_end = EVAL_EPSILON
    agent.steps_done = 1000000  # Force epsilon to end value
    
    print(f"\nWatching {num_episodes} episodes...")
    print("Press 'q' in game window or Ctrl+C to stop")
    print("-" * 60)
    
    # Create game display window
    cv2.namedWindow("Pong AI - Watch Mode", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pong AI - Watch Mode", 640, 480)
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    recent_rewards = deque(maxlen=20)
    recent_lengths = deque(maxlen=20)  # For Steps Avg(20)
    
    start_time = time.time()
    
    try:
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            
            # DeepMind: 0-30 random no-ops at start to preventing overfitting to start state
            no_ops = np.random.randint(0, 31)
            for _ in range(no_ops):
                # Action 0 = NOOP
                state, _, done, _, _ = env.step(0)
                if done: state = env.reset()
                
            episode_reward = 0
            steps = 0
            
            while True:
                # Select action with eval epsilon
                action = agent.select_action(state, training=True)  # training=True to use epsilon
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                # Display game frame and check for quit
                try:
                    frame = env.env.render()
                    if frame is not None:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # Add HUD (Removed per user request)
                        # cv2.putText(frame_bgr, f"Episode: {episode}/{num_episodes}", 
                        #             (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        # cv2.putText(frame_bgr, f"Reward: {episode_reward:+.0f}", 
                        #             (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        # cv2.putText(frame_bgr, f"Steps: {steps}", 
                        #             (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        # cv2.putText(frame_bgr, "Press 'q' to quit",
                        #             (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
                        cv2.imshow("Pong AI - Watch Mode", frame_bgr)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            print("\nQuitting (q key pressed)...")
                            raise KeyboardInterrupt
                except KeyboardInterrupt:
                    raise  # Re-raise to exit
                except Exception:
                    pass  # Ignore render errors only
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            recent_rewards.append(episode_reward)
            recent_lengths.append(steps)
            avg_reward = np.mean(recent_rewards)
            avg_steps = np.mean(recent_lengths)
            
            result = "WIN" if episode_reward > 0 else "LOSS" if episode_reward < -20 else "CLOSE"
            elapsed = time.time() - start_time
            print(f"Episode {episode:3d} | Reward: {episode_reward:+6.0f} | "
                  f"Avg(20): {avg_reward:+6.1f} | Steps Avg(20): {avg_steps:5.0f} | "
                  f"Result: {result} | Time: {elapsed/60:.1f}m")
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    cv2.destroyAllWindows()
    env.close()
    
    # Save evaluation graph
    if episode_rewards:
        graph_path = save_evaluation_graph(episode_rewards, episode_lengths, checkpoint_dir, "watch")
        print_summary(episode_rewards, episode_lengths, graph_path)


def benchmark_agent(
    checkpoint_path: str = "checkpoints/pong_dqn_final.pt",
    num_episodes: int = 100,
    checkpoint_dir: str = "checkpoints",
    record: bool = False
):
    """
    Background evaluation with terminal output only (fast, no display).
    Generates performance graphs at the end.
    
    Args:
        record: If True, record frames and export video (requires rgb_array render mode)
    """
    print("=" * 60)
    print("DQN Pong - Benchmark Mode (Background)")
    print("=" * 60)
    print(f"Evaluation epsilon: {EVAL_EPSILON}")
    print(f"Evaluation conditions: 0-30 random no-ops at start")
    if record:
        print("[RECORDING] Capturing frames for video export")
    else:
        print("[NO DISPLAY] Running in headless mode for speed")
    
    # Create environment - need rgb_array if recording
    render_mode = "rgb_array" if record else None
    env = PongEnvironment(render_mode=render_mode)
    
    # Setup recorder if enabled
    recorder = None
    if record:
        eval_frames_dir = os.path.join(checkpoint_dir, "evaluation_frames")
        os.makedirs(eval_frames_dir, exist_ok=True)
        recorder = TrainingRecorder(
            checkpoint_dir=checkpoint_dir,
            frame_skip=2,  # Every 2nd frame for evaluation (faster than training)
            buffer_size=300
        )
        # Override paths to use evaluation-specific directories
        recorder.frames_dir = eval_frames_dir
        recorder.state_path = os.path.join(checkpoint_dir, "evaluation_state.json")
        recorder.start()
    
    # Create and load agent
    agent = DQNAgent(n_actions=env.n_actions, device="cpu")
    
    try:
        agent.load(checkpoint_path)

        print(f"Agent has {agent.steps_done:,} training steps")
    except FileNotFoundError:
        print(f"No checkpoint found at {checkpoint_path}")
        print("Using random agent...")
    
    # Force evaluation epsilon
    agent.epsilon_start = EVAL_EPSILON
    agent.epsilon_end = EVAL_EPSILON
    agent.steps_done = 1000000
    
    print(f"\nRunning {num_episodes} episodes...")
    print("-" * 60)
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    recent_rewards = deque(maxlen=20)
    recent_lengths = deque(maxlen=20)  # For Steps Avg(20)
    
    start_time = time.time()
    
    try:
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            
            # DeepMind: 0-30 random no-ops at start
            no_ops = np.random.randint(0, 31)
            for _ in range(no_ops):
                state, _, done, _, _ = env.step(0)
                if done: state = env.reset()
                
            episode_reward = 0
            steps = 0
            
            while True:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                # Record frame if enabled
                if recorder is not None:
                    frame = env.render()
                    if frame is not None:
                        recorder.add_frame(frame)
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            recent_rewards.append(episode_reward)
            recent_lengths.append(steps)
            avg_reward = np.mean(recent_rewards)
            avg_steps = np.mean(recent_lengths)
            
            # Update recorder graph if enabled
            if recorder is not None:
                recorder.update_graph(episode, episode_reward, avg_reward, EVAL_EPSILON, steps)
            
            # Print every 5 episodes
            if episode % 5 == 0:
                elapsed = time.time() - start_time
                print(f"Episode {episode:3d} | Reward: {episode_reward:+6.0f} | "
                      f"Avg(20): {avg_reward:+6.1f} | Steps Avg(20): {avg_steps:5.0f} | "
                      f"Time: {elapsed/60:.1f}m")
            

                
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    env.close()
    
    # Save recorder state and close if enabled
    if recorder is not None:
        recorder.save()
        recorder.close()
        print(f"\nRecorder: Frames saved to {recorder.frames_dir}")
        print(f"Use 'python main.py export-video' to create video")
    
    # Save evaluation graph
    if episode_rewards:
        graph_path = save_evaluation_graph(episode_rewards, episode_lengths, checkpoint_dir, "benchmark")
        print_summary(episode_rewards, episode_lengths, graph_path)


def record_evaluation_episode(
    checkpoint_path: str = "checkpoints/pong_dqn_final.pt",
    checkpoint_dir: str = "checkpoints"
):
    """
    Record a single high-quality episode for video exhibition.
    Settings: epsilon=0.0, 30 random no-ops at start, frame_skip=1.
    """
    print("=" * 60)
    print("DQN Pong - Record Evaluation Episode")
    print("=" * 60)
    print(f"Settings: Epsilon=0.0, No-ops=30, Quality=Max (skip=1)")
    
    # Create environment with RGB output
    env = PongEnvironment(render_mode="rgb_array")
    
    # Setup recorder with frame_skip=1 for highest quality
    eval_frames_dir = os.path.join(checkpoint_dir, "evaluation_frames")
    
    # Try to clean up old evaluation frames - use a more robust approach for Windows
    if os.path.exists(eval_frames_dir):
        try:
            import shutil
            # On Windows, rmtree often fails if explorer or another process has a handle
            # We'll try to rename first or ignore errors, then recreate
            shutil.rmtree(eval_frames_dir, ignore_errors=True)
        except Exception:
            pass
    
    os.makedirs(eval_frames_dir, exist_ok=True)
    
    recorder = TrainingRecorder(
        checkpoint_dir=checkpoint_dir,
        frame_skip=1,  # Maximum quality: record every single frame
        buffer_size=500
    )
    recorder.frames_dir = eval_frames_dir
    recorder.state_path = os.path.join(checkpoint_dir, "evaluation_state.json")
    recorder.start()
    
    # Create and load agent
    agent = DQNAgent(n_actions=env.n_actions, device="cpu")
    
    try:
        agent.load(checkpoint_path)
    except FileNotFoundError:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # Force evaluation epsilon
    agent.epsilon_start = EVAL_EPSILON
    agent.epsilon_end = EVAL_EPSILON
    agent.steps_done = 1000000
    
    num_episodes = 5
    print(f"\nPlaying {num_episodes} episodes...")
    
    start_time = time.time()
    
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        
        # Fixed 30 random no-ops as requested
        for _ in range(30):
            state, _, done, _, _ = env.step(0)
            if done: state = env.reset()
            
        episode_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Record every frame
            frame = env.render()
            if frame is not None:
                recorder.add_frame(frame)
            
            if done:
                break
        
        print(f"Episode {ep} finished | Reward: {episode_reward:+6.0f} | Steps: {steps}")
        # Update recorder with data for the graph (even if not shown in this layout)
        recorder.update_graph(ep, episode_reward, episode_reward, EVAL_EPSILON, steps)
    
    elapsed = time.time() - start_time
    print(f"\nAll episodes finished | Total Time: {elapsed:.1f}s")
    
    env.close()
    
    # Save and Export
    recorder.save()
    output_path = os.path.join(checkpoint_dir, "evaluation_recording.mp4")
    print("\nExporting high-quality video (Game Only)...")
    video_path = recorder.export_video(output_path=output_path, fps=30, quality="high", layout="game-only")
    recorder.close()
    
    print("=" * 60)
    print(f"SUCCESS: High-quality recording saved to: {video_path}")
    print("=" * 60)


def save_evaluation_graph(rewards, lengths, save_dir, mode_name):
    """Save evaluation performance graphs."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='#1a1a1a')
    
    # Rewards plot
    axes[0].set_facecolor('#1a1a1a')
    axes[0].plot(rewards, 'cyan', alpha=0.5, label='Episode Reward')
    
    # Rolling average
    if len(rewards) >= 10:
        rolling = np.convolve(rewards, np.ones(10)/10, mode='valid')
        axes[0].plot(range(9, len(rewards)), rolling, 'lime', linewidth=2, label='10-Episode Avg')
    
    axes[0].axhline(y=0, color='white', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Episode', color='white')
    axes[0].set_ylabel('Reward', color='white')
    axes[0].set_title(f'Evaluation Results', fontsize=14, fontweight='bold', color='white')
    legend = axes[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15), 
                           ncol=2, facecolor='#2a2a2a', edgecolor='gray')
    plt.setp(legend.get_texts(), color='white')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(colors='white')
    
    # Episode lengths plot
    axes[1].set_facecolor('#1a1a1a')
    axes[1].plot(lengths, 'orange', alpha=0.5, label='Steps per Episode')
    
    if len(lengths) >= 10:
        rolling_len = np.convolve(lengths, np.ones(10)/10, mode='valid')
        axes[1].plot(range(9, len(lengths)), rolling_len, 'yellow', linewidth=2, label='10-Episode Avg')
    
    axes[1].set_xlabel('Episode', color='white')
    axes[1].set_ylabel('Steps', color='white')
    axes[1].set_title('Episode Lengths', fontsize=14, fontweight='bold', color='white')
    legend = axes[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.15),
                           ncol=2, facecolor='#2a2a2a', edgecolor='gray')
    plt.setp(legend.get_texts(), color='white')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(colors='white')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'evaluation_{mode_name}.png')
    plt.savefig(save_path, dpi=150, facecolor='#1a1a1a')
    plt.close()
    return save_path


def print_summary(rewards, lengths, graph_path=None):
    """Print evaluation summary statistics."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if graph_path:
        print(f"Evaluation graph saved to: {graph_path}\n")
        
    print(f"Episodes completed: {len(rewards)}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Best reward: {max(rewards):.0f}")
    print(f"Worst reward: {min(rewards):.0f}")
    print(f"Std dev: {np.std(rewards):.2f}")
    print(f"Average steps: {np.mean(lengths):.0f}")
    
    wins = sum(1 for r in rewards if r > 0)
    close_games = sum(1 for r in rewards if -10 <= r <= 0)
    losses = sum(1 for r in rewards if r < -10)
    
    print(f"\nResults: {wins} wins, {close_games} close games, {losses} losses")
    print(f"Win rate: {wins / len(rewards) * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained DQN Pong agent")
    parser.add_argument("mode", choices=["watch", "benchmark"], 
                        help="Evaluation mode")
    parser.add_argument("--checkpoint", "-c", type=str, 
                        default="checkpoints/pong_dqn_final.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", "-e", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--checkpoint-dir", "-d", type=str, default="checkpoints",
                        help="Directory to save evaluation graphs")
    parser.add_argument("--record", "-r", action="store_true",
                        help="Record frames for video export (benchmark mode only)")
    
    args = parser.parse_args()
    
    if args.mode == "watch":
        watch_agent(args.checkpoint, args.episodes, args.checkpoint_dir)
    elif args.mode == "benchmark":
        benchmark_agent(args.checkpoint, args.episodes, args.checkpoint_dir, args.record)
