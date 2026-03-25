"""
Pong AI - Main Entry Point

A Deep Q-Network (DQN) agent that learns to play Atari Pong.

Usage:
    python main.py train        - Train a new agent
    python main.py train-visual - Train with live game/graph windows
    python main.py play         - Watch trained agent play
    python main.py watch        - Watch agent with visual graph (epsilon=0.01)
    python main.py benchmark    - Fast background evaluation with stats
    python main.py record-eval  - Record high-quality video of 5 episodes (game-only)
    python main.py test         - Test environment setup
    python main.py reset        - Delete all training data
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="DQN Pong AI - Self-learning agent for Atari Pong"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the DQN agent")
    train_parser.add_argument("--episodes", "-e", type=int, default=500,
                              help="Number of training episodes (default: 500)")
    train_parser.add_argument("--checkpoint-dir", "-d", type=str, default="checkpoints",
                              help="Directory to save checkpoints")
    train_parser.add_argument("--record", "-r", action="store_true",
                              help="Enable video recording of training session")
    
    # Train-visual command (with live graph and game window)
    train_visual_parser = subparsers.add_parser("train-visual", 
                                                 help="Train with live game window and performance graph")
    train_visual_parser.add_argument("--episodes", "-e", type=int, default=500,
                                      help="Number of training episodes (default: 500)")
    train_visual_parser.add_argument("--checkpoint-dir", "-d", type=str, default="checkpoints",
                                      help="Directory to save checkpoints")
    train_visual_parser.add_argument("--record", "-r", action="store_true",
                                      help="Enable video recording of training session")
    
    # Play command
    play_parser = subparsers.add_parser("play", help="Watch trained agent play")
    play_parser.add_argument("--checkpoint", "-c", type=str, 
                             default="checkpoints/pong_dqn_best.pt",
                             help="Path to model checkpoint")
    play_parser.add_argument("--episodes", "-e", type=int, default=5,
                             help="Number of episodes to play")
    
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test environment setup")
    
    # Watch command - visual evaluation with graph (no training)
    watch_parser = subparsers.add_parser("watch", 
                                          help="Watch trained agent play with live graph (epsilon=0.01)")
    watch_parser.add_argument("--checkpoint", "-c", type=str, 
                              default="checkpoints/pong_dqn_final.pt",
                              help="Path to model checkpoint")
    watch_parser.add_argument("--episodes", "-e", type=int, default=10,
                              help="Number of episodes to watch")
    watch_parser.add_argument("--checkpoint-dir", "-d", type=str, default="checkpoints",
                              help="Directory to save evaluation graphs")
    
    # Benchmark command - background evaluation (terminal only, fast)
    benchmark_parser = subparsers.add_parser("benchmark", 
                                              help="Background evaluation with terminal output (epsilon=0.01)")
    benchmark_parser.add_argument("--checkpoint", "-c", type=str, 
                                   default="checkpoints/pong_dqn_final.pt",
                                   help="Path to model checkpoint")
    benchmark_parser.add_argument("--episodes", "-e", type=int, default=100,
                                   help="Number of episodes to evaluate")
    benchmark_parser.add_argument("--checkpoint-dir", "-d", type=str, default="checkpoints",
                                   help="Directory to save evaluation graphs")
    benchmark_parser.add_argument("--record", "-r", action="store_true",
                                   help="Record frames for video export")
    
    # Export training video command
    export_train_parser = subparsers.add_parser("export-training-video", 
                                                 help="Export recorded training frames to MP4 video")
    export_train_parser.add_argument("--checkpoint-dir", "-d", type=str, default="checkpoints",
                                     help="Directory containing training frames")
    export_train_parser.add_argument("--output", "-o", type=str, default=None,
                                     help="Output video path (default: checkpoints/training_recording.mp4)")
    export_train_parser.add_argument("--fps", type=int, default=30,
                                     help="Video frames per second (default: 30)")
    export_train_parser.add_argument("--quality", "-q", type=str, default="high",
                                     choices=["high", "medium", "low"],
                                     help="Video quality: high (full res), medium (75%%), low (50%%)")
    export_train_parser.add_argument("--layout", "-l", type=str, default="side-by-side",
                                     choices=["side-by-side", "game-only", "graph-only"],
                                     help="Video layout: side-by-side, game-only, or graph-only")
    export_train_parser.add_argument("--limit", type=int, default=None,
                                     help="Only export the last N frames (prevents massive videos)")
    
    # Export benchmark video command
    export_bench_parser = subparsers.add_parser("export-benchmark-video", 
                                                 help="Export recorded benchmark frames to MP4 video")
    export_bench_parser.add_argument("--checkpoint-dir", "-d", type=str, default="checkpoints",
                                     help="Directory containing benchmark frames")
    export_bench_parser.add_argument("--output", "-o", type=str, default=None,
                                     help="Output video path (default: checkpoints/benchmark_recording.mp4)")
    export_bench_parser.add_argument("--fps", type=int, default=30,
                                     help="Video frames per second (default: 30)")
    export_bench_parser.add_argument("--quality", "-q", type=str, default="high",
                                     choices=["high", "medium", "low"],
                                     help="Video quality: high (full res), medium (75%%), low (50%%)")
    export_bench_parser.add_argument("--layout", "-l", type=str, default="side-by-side",
                                     choices=["side-by-side", "game-only", "graph-only"],
                                     help="Video layout: side-by-side, game-only, or graph-only")
    export_bench_parser.add_argument("--limit", type=int, default=None,
                                     help="Only export the last N frames")
    
    # Reset command - delete training/evaluation data
    reset_parser = subparsers.add_parser("reset", 
                                          help="Delete training and/or evaluation data")
    reset_parser.add_argument("--checkpoint-dir", "-d", type=str, default="checkpoints",
                              help="Directory to clear (default: checkpoints)")
    reset_parser.add_argument("--training", "-t", action="store_true",
                              help="Reset training data only (frames, checkpoints, state)")
    reset_parser.add_argument("--benchmark", "-b", action="store_true",
                              help="Reset benchmark/evaluation data only")
    reset_parser.add_argument("--video", "-v", action="store_true",
                              help="Reset video frames only (training_frames + evaluation_frames)")
    reset_parser.add_argument("--all", "-a", action="store_true",
                              help="Reset all data (training + benchmark)")
    reset_parser.add_argument("--yes", "-y", action="store_true",
                              help="Skip confirmation prompt")
    
    # Record-eval command - high quality 5-episode recording
    record_eval_parser = subparsers.add_parser("record-eval", 
                                                help="Record high-quality video of 5 episodes (game-only)")
    record_eval_parser.add_argument("--checkpoint", "-c", type=str, 
                                     default="checkpoints/pong_dqn_final.pt",
                                     help="Path to model checkpoint")
    record_eval_parser.add_argument("--checkpoint-dir", "-d", type=str, default="checkpoints",
                                     help="Directory to save evaluation data")
    
    args = parser.parse_args()
    
    if args.command == "train":
        from train import train
        train(
            num_episodes=args.episodes,
            checkpoint_dir=args.checkpoint_dir,
            record=args.record
        )
    
    elif args.command == "train-visual":
        from train_visual import train_visual
        train_visual(
            num_episodes=args.episodes,
            checkpoint_dir=args.checkpoint_dir,
            record=args.record
        )
        
    elif args.command == "play":
        from play import play
        play(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes
        )
    
    
    elif args.command == "watch":
        from evaluate import watch_agent
        watch_agent(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            checkpoint_dir=args.checkpoint_dir
        )
    
    elif args.command == "benchmark":
        from evaluate import benchmark_agent
        benchmark_agent(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            checkpoint_dir=args.checkpoint_dir,
            record=args.record
        )
    
    elif args.command == "record-eval":
        from evaluate import record_evaluation_episode
        record_evaluation_episode(
            checkpoint_path=args.checkpoint,
            checkpoint_dir=args.checkpoint_dir
        )
    
    elif args.command == "export-training-video":
        from recorder import export_training_video
        result = export_training_video(
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output,
            fps=args.fps,
            quality=args.quality,
            layout=args.layout,
            limit=args.limit
        )
        if result:
            print(f"\nTraining video ready: {result}")
        else:
            print("\nNo video exported. Make sure you have run training with --record first.")
    
    elif args.command == "export-benchmark-video":
        from recorder import TrainingRecorder
        import os
        
        # Create recorder pointing to evaluation frames
        eval_frames_dir = os.path.join(args.checkpoint_dir, "evaluation_frames")
        if not os.path.exists(eval_frames_dir):
            print(f"No benchmark frames found at {eval_frames_dir}")
            print("Run 'python main.py benchmark --record' first.")
        else:
            recorder = TrainingRecorder(checkpoint_dir=args.checkpoint_dir)
            recorder.frames_dir = eval_frames_dir
            recorder.state_path = os.path.join(args.checkpoint_dir, "evaluation_state.json")
            recorder.start()
            
            output_path = args.output or os.path.join(args.checkpoint_dir, "benchmark_recording.mp4")
            result = recorder.export_video(
                output_path=output_path,
                fps=args.fps,
                quality=args.quality,
                layout=args.layout,
                limit=args.limit
            )
            recorder.close()
            
            if result:
                print(f"\nBenchmark video ready: {result}")
            else:
                print("\nNo video exported.")
    
    elif args.command == "reset":
        import os
        import shutil
        
        checkpoint_dir = args.checkpoint_dir
        
        # Determine what to reset
        reset_training = args.training or args.all
        reset_benchmark = args.benchmark or args.all
        reset_video = args.video
        
        # If no specific flag, show help
        if not (args.training or args.benchmark or args.video or args.all):
            print("Please specify what to reset:")
            print("  --training, -t   Reset training data only")
            print("  --benchmark, -b  Reset benchmark/evaluation data only")
            print("  --video, -v      Reset video frames only (preserves checkpoints)")
            print("  --all, -a        Reset all data (training + benchmark)")
            print("\nExample: python main.py reset --video")
            return
        
        # Check if directory exists
        if not os.path.exists(checkpoint_dir):
            print(f"Nothing to reset - '{checkpoint_dir}' directory doesn't exist.")
            return
        
        # Define what belongs to training vs benchmark vs video-only
        training_items = {
            'training_frames',       # Directory
            'training_state.json',   # Training recorder state
            'training_recording.mp4',
            'training_progress.png',
        }
        training_patterns = ['pong_dqn_']  # Checkpoint files
        
        benchmark_items = {
            'evaluation_frames',     # Directory
            'evaluation_state.json', # Benchmark recorder state
            'benchmark_recording.mp4',
            'evaluation_benchmark.png',
            'evaluation_watch.png',
        }
        
        # Video-only: just the frame directories and state files, no checkpoints
        video_only_items = {
            'training_frames',
            'training_state.json',
            'training_recording.mp4',
            'evaluation_frames',
            'evaluation_state.json',
            'evaluation_recording.mp4',
            'benchmark_recording.mp4',
            'evaluation_benchmark.png',
            'evaluation_watch.png',
        }
        
        # Collect items to delete
        items_to_delete = []
        
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            
            # Determine category
            is_training = item in training_items or any(item.startswith(p) for p in training_patterns)
            is_benchmark = item in benchmark_items
            is_video_only = item in video_only_items
            
            # Determine if should delete based on flags
            if reset_video:
                should_delete = is_video_only
            else:
                should_delete = (reset_training and is_training) or (reset_benchmark and is_benchmark)
            
            if should_delete:
                if os.path.isdir(item_path):
                    items_to_delete.append((item_path, f"  [DIR] {item}/"))
                else:
                    size_mb = os.path.getsize(item_path) / (1024 * 1024)
                    items_to_delete.append((item_path, f"  [FILE] {item} ({size_mb:.1f} MB)"))
        
        if not items_to_delete:
            reset_type = "video" if reset_video else "training" if reset_training and not reset_benchmark else "benchmark" if reset_benchmark and not reset_training else "all"
            print(f"Nothing to reset - no {reset_type} data found in '{checkpoint_dir}'.")
            return
        
        # Show what will be deleted
        reset_desc = []
        if reset_video:
            reset_desc.append("VIDEO")
        else:
            if reset_training:
                reset_desc.append("TRAINING")
            if reset_benchmark:
                reset_desc.append("BENCHMARK")
        
        print(f"\n[DELETE] The following {' + '.join(reset_desc)} items will be DELETED:\n")
        for _, display in items_to_delete:
            print(display)
        print()
        
        # Confirm unless --yes flag is used
        if not args.yes:
            response = input("Are you sure? Type 'yes' to confirm: ")
            if response.lower() != 'yes':
                print("Reset cancelled.")
                return
        
        # Delete items
        deleted_count = 0
        error_count = 0
        
        for item_path, display in items_to_delete:
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
                print(f"  [DELETED] {os.path.basename(item_path)}")
                deleted_count += 1
            except Exception as e:
                print(f"  [ERROR] Could not delete {os.path.basename(item_path)}: {e}")
                error_count += 1
        
        if error_count > 0:
            print(f"\n[WARNING] Reset completed with {error_count} errors.")
        else:
            print(f"\n[DONE] {' + '.join(reset_desc)} data deleted! ({deleted_count} items removed)")
        

        
    elif args.command == "test":
        print("Testing environment setup...")
        
        # Test imports
        print("\n1. Testing imports...")
        try:
            import torch
            print(f"   [OK] PyTorch {torch.__version__}")
        except ImportError as e:
            print(f"   [X] PyTorch: {e}")
            return
            
        try:
            import gymnasium
            print(f"   [OK] Gymnasium {gymnasium.__version__}")
        except ImportError as e:
            print(f"   [X] Gymnasium: {e}")
            return
            
        try:
            import cv2
            print(f"   [OK] OpenCV {cv2.__version__}")
        except ImportError as e:
            print(f"   [X] OpenCV: {e}")
            return
            
        try:
            import numpy
            print(f"   [OK] NumPy {numpy.__version__}")
        except ImportError as e:
            print(f"   [X] NumPy: {e}")
            return
        
        # Test Atari environment
        print("\n2. Testing Atari environment...")
        try:
            from environment import PongEnvironment
            env = PongEnvironment(render_mode=None)
            state = env.reset()
            print(f"   [OK] Environment created")
            print(f"   [OK] State shape: {state.shape}")
            print(f"   [OK] Actions: {env.n_actions}")
            
            # Take a step
            next_state, reward, done, truncated, info = env.step(0)
            print(f"   [OK] Step successful")
            env.close()
            
        except Exception as e:
            print(f"   [X] Environment error: {e}")
            print("\n   Hint: Make sure you ran 'AutoROM --accept-license'")
            return
        
        # Test model
        print("\n3. Testing model...")
        try:
            from model import DQN
            model = DQN(n_actions=3)
            dummy_input = torch.randn(1, 4, 84, 84)
            output = model(dummy_input)
            print(f"   [OK] Model created")
            print(f"   [OK] Output shape: {output.shape}")
        except Exception as e:
            print(f"   [X] Model error: {e}")
            return
        
        # Test agent
        print("\n4. Testing agent...")
        try:
            from agent import DQNAgent
            agent = DQNAgent(n_actions=3)
            state = numpy.random.rand(4, 84, 84).astype(numpy.float32)
            action = agent.select_action(state)
            print(f"   [OK] Agent created")
            print(f"   [OK] Action selection works: action={action}")
        except Exception as e:
            print(f"   [X] Agent error: {e}")
            return
        
        print("\n" + "=" * 50)
        print("All tests passed! You're ready to train.")
        print("Run: python main.py train")
        print("=" * 50)
        
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. python main.py test         - Verify setup")
        print("  2. python main.py train        - Train the agent")
        print("  3. python main.py play         - Watch it play!")
        print("  4. python main.py export-training-video  - Export training recording")
        print("  5. python main.py export-benchmark-video - Export benchmark recording")
        print("  6. python main.py reset --training       - Reset training data")
        print("  7. python main.py reset --benchmark      - Reset benchmark data")


if __name__ == "__main__":
    main()
