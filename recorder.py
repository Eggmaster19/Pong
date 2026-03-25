"""
Training Recorder - Captures training sessions as video.
Renders Pong game on the left and live performance graph on the right.
Supports pause/resume across training sessions.
"""

import os
import json
import queue
import numpy as np
import cv2
import threading
from typing import Optional, List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Headless backend - no display needed
import matplotlib.pyplot as plt


class TrainingRecorder:
    """
    Records training sessions with game visualization and performance graph.
    
    Features:
    - Captures game frames during training
    - Generates live-updating graph of rewards and epsilon
    - Saves frames to disk for resume capability
    - Exports final video with game on left, graph on right
    """
    
    # Video dimensions
    GAME_WIDTH = 640
    GAME_HEIGHT = 720  # Expanded vertically for 16:9 total
    GRAPH_WIDTH = 640
    GRAPH_HEIGHT = 720 # Expanded vertically
    COMBINED_WIDTH = GAME_WIDTH + GRAPH_WIDTH  # 1280
    COMBINED_HEIGHT = 720 # 1280x720 is exactly 16:9
    
    # Frame storage: organize into subdirectories to avoid NTFS slowdown
    FRAMES_PER_BATCH = 1000  # Keep each directory under 1000 files
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        frame_skip: int = 4,  # Save every Nth frame to manage disk space
        buffer_size: int = 300,  # Number of frames to buffer in RAM before flushing
    ):
        """
        Initialize the training recorder.
        
        Args:
            checkpoint_dir: Directory for saving frames and state
            frame_skip: Save every Nth frame (reduces disk usage)
            buffer_size: Number of frames to hold in RAM before writing to disk
        """
        self.checkpoint_dir = checkpoint_dir
        self.frame_skip = frame_skip
        self.buffer_size = buffer_size
        
        # Paths
        self.frames_dir = os.path.join(checkpoint_dir, "training_frames")
        self.state_path = os.path.join(checkpoint_dir, "training_state.json")
        
        # Graph data
        self.episodes: List[int] = []
        self.rewards: List[float] = []
        self.avg_rewards: List[float] = []
        self.epsilons: List[float] = []
        self.episode_lengths: List[int] = []
        self.avg_lengths: List[float] = []
        
        # Frame tracking
        self.frame_count = 0
        self.step_count = 0
        self.total_steps = 0
        
        # Queue-based async writer (bounded to prevent RAM exhaustion)
        # maxsize=buffer_size caps RAM at ~800MB (300 frames × 2.64MB each)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_writer = threading.Event()
        
        # Graph caching (Optimization: only render when data changes)
        self.last_graph_frame = None
        
        # Setup matplotlib figure for graph (headless)
        self._setup_graph()
        
        self.initialized = False
        
    def _setup_graph(self):
        """Setup the matplotlib figure for the performance graph."""
        plt.style.use('dark_background')
        # figsize(6.4, 7.2) at 100dpi = 640x720 pixels
        self.fig, self.axes = plt.subplots(2, 1, figsize=(6.4, 7.2), dpi=100)
        self.fig.patch.set_facecolor('#1a1a1a')
        
        # Reward plot
        self.ax_reward = self.axes[0]
        self.ax_reward.set_facecolor('#1a1a1a')
        self.ax_reward.set_xlabel('Episode', fontsize=9)
        self.ax_reward.set_ylabel('Reward', fontsize=9)
        self.ax_reward.set_title('Training Rewards', fontsize=11, fontweight='bold')
        self.ax_reward.grid(True, alpha=0.3)
        self.ax_reward.set_xlim(0, 50)
        self.ax_reward.set_ylim(-22, 22)
        
        self.line_reward, = self.ax_reward.plot([], [], 'cyan', alpha=0.6, 
                                                  linewidth=1.5, marker='o', 
                                                  markersize=2, label='Episode Reward')
        self.line_avg, = self.ax_reward.plot([], [], 'lime', linewidth=2.5, 
                                               label='20-Episode Avg')
        # self.ax_reward.legend(loc='upper right', bbox_to_anchor=(1.0, 1.35), 
        #                      ncol=2, fontsize=8)
        
        # Epsilon plot
        # Steps plot
        self.ax_steps = self.axes[1]
        self.ax_steps.set_facecolor('#1a1a1a')
        self.ax_steps.set_xlabel('Episode', fontsize=9)
        self.ax_steps.set_ylabel('Steps', fontsize=9, color='orange')
        self.ax_steps.set_title('Episode Length', fontsize=11, fontweight='bold')
        self.ax_steps.grid(True, alpha=0.3)
        self.ax_steps.set_xlim(0, 50)
        self.ax_steps.set_ylim(0, 5000)  # Will auto-scale
        
        self.line_steps, = self.ax_steps.plot([], [], 'orange', alpha=0.6, linewidth=1.5, label='Steps')
        self.line_avg_steps, = self.ax_steps.plot([], [], 'yellow', linewidth=2.5, label='20-Ep Avg')
        # self.ax_steps.legend(loc='upper right', bbox_to_anchor=(1.0, 1.35),
        #                     ncol=2, fontsize=8)
        
        # Add Epsilon Text Annotation in figure coords (bottom left)
        self.epsilon_text = self.fig.text(0.02, 0.02, '', color='orange', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
    def start(self):
        """
        Start or resume recording.
        Loads existing state if available.
        """
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Try to load existing state
        if os.path.exists(self.state_path):
            self._load_state()
            print(f"Recorder: Resumed with {self.frame_count} frames, {len(self.episodes)} episodes")
        else:
            print("Recorder: Starting fresh recording")
        
        # Start the background writer thread
        self._stop_writer.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
            
        self.initialized = True
        
    def _load_state(self):
        """Load saved state from disk."""
        with open(self.state_path, 'r') as f:
            state = json.load(f)
            
        self.episodes = state.get('episodes', [])
        self.rewards = state.get('rewards', [])
        self.avg_rewards = state.get('avg_rewards', [])
        self.epsilons = state.get('epsilons', [])
        # Backwards compatibility: load if exists, else init empty
        self.episode_lengths = state.get('episode_lengths', [])
        self.avg_lengths = state.get('avg_lengths', [])
        self.frame_count = state.get('frame_count', 0)
        self.total_steps = state.get('total_steps', 0)
        
        # Update graph with loaded data
        if self.episodes:
            self._update_graph_lines()
            
    def save(self):
        """Save current state to disk."""
        state = {
            'episodes': self.episodes,
            'rewards': self.rewards,
            'avg_rewards': self.avg_rewards,
            'epsilons': self.epsilons,
            'episode_lengths': self.episode_lengths,
            'avg_lengths': self.avg_lengths,
            'frame_count': self.frame_count,
            'total_steps': self.total_steps,
        }
        
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        print(f"Recorder: State saved ({self.frame_count} frames, {len(self.episodes)} episodes)")
        
    def add_frame(self, game_frame: np.ndarray):
        """
        Add a game frame to the recording.
        Frames are buffered in RAM and flushed to disk asynchronously.
        
        Args:
            game_frame: RGB frame from the game (any size, will be resized)
        """
        if not self.initialized:
            return
            
        self.step_count += 1
        self.total_steps += 1
        
        # Only save every Nth frame
        if self.step_count % self.frame_skip != 0:
            return
            
        # Resize game frame
        if game_frame.shape[:2] != (self.GAME_HEIGHT, self.GAME_WIDTH):
            game_resized = cv2.resize(game_frame, (self.GAME_WIDTH, self.GAME_HEIGHT))
        else:
            game_resized = game_frame
            
        # Ensure RGB format
        if len(game_resized.shape) == 2:  # Grayscale
            game_resized = cv2.cvtColor(game_resized, cv2.COLOR_GRAY2RGB)
        elif game_resized.shape[2] == 4:  # RGBA
            game_resized = cv2.cvtColor(game_resized, cv2.COLOR_RGBA2RGB)
            
        # Render current graph to numpy array (or use cache)
        if self.last_graph_frame is None:
            self.last_graph_frame = self._render_graph()
        graph_frame = self.last_graph_frame
        
        # Combine horizontally: game on left, graph on right
        combined = np.hstack([game_resized, graph_frame])
        
        # Add to queue (blocks if queue is full - provides backpressure)
        # This caps RAM usage at buffer_size × ~2.64MB per frame
        self.frame_count += 1
        self._frame_queue.put((self.frame_count, combined.copy()))  # Blocking put
    
    def _get_frame_path(self, frame_num: int) -> str:
        """
        Get the full path for a frame, organized into batch subdirectories.
        This keeps each directory under FRAMES_PER_BATCH files for NTFS performance.
        """
        batch_num = (frame_num - 1) // self.FRAMES_PER_BATCH
        batch_dir = os.path.join(self.frames_dir, f"batch_{batch_num:04d}")
        os.makedirs(batch_dir, exist_ok=True)
        return os.path.join(batch_dir, f"frame_{frame_num:08d}.jpg")
    
    def _writer_loop(self):
        """
        Background writer thread. Continuously pulls frames from queue and writes to disk.
        Runs independently of training - never blocks the main thread.
        Uses JPEG encoding for ~10x faster compression than PNG.
        Frames are organized into batch subdirectories for NTFS performance.
        """
        while not self._stop_writer.is_set() or not self._frame_queue.empty():
            try:
                # Wait for a frame with timeout (allows checking stop signal)
                frame_num, frame_data = self._frame_queue.get(timeout=0.1)
                frame_path = self._get_frame_path(frame_num)
                # JPEG quality 90 = good quality, ~10x faster than PNG
                cv2.imwrite(frame_path, cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR), 
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
                self._frame_queue.task_done()
            except queue.Empty:
                continue  # No frames available, loop again
    
    def flush_remaining(self):
        """
        Wait for all queued frames to be written. Call before closing.
        This is a blocking call to ensure all frames are saved.
        """
        # Signal writer to stop after draining queue
        self._stop_writer.set()
        
        # Wait for writer thread to finish
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=60)  # Wait up to 60s for remaining writes
            
        # If thread is still alive, warn user
        if self._writer_thread is not None and self._writer_thread.is_alive():
            print(f"Recorder: WARNING - Writer thread still has {self._frame_queue.qsize()} frames pending")
        
    def update_graph(self, episode: int, reward: float, avg_reward: float, epsilon: float, steps: int = None):
        """
        Update the graph with new episode data.
        
        Args:
            episode: Episode number (ignored - we use sequential internal numbering)
            reward: Episode reward
            reward: Episode reward
            avg_reward: Moving average reward
            epsilon: Current epsilon value
            steps: Number of steps in the episode
        """
        # Always use sequential episode numbering to avoid gaps in the graph
        # This ensures continuous display even if training restarts with different episode numbers
        next_episode = len(self.episodes) + 1
        
        self.episodes.append(next_episode)
        self.rewards.append(reward)
        self.avg_rewards.append(avg_reward)
        self.epsilons.append(epsilon)
        
        # Invalidate graph cache since data changed
        self.last_graph_frame = None
        
        # Calculate moving average for steps
        if steps is not None:
            self.episode_lengths.append(steps)
            # Calculate 20-episode average
            recent_steps = self.episode_lengths[-20:]
            avg_steps = sum(recent_steps) / len(recent_steps)
            self.avg_lengths.append(avg_steps)
        else:
            # Fallback if steps not provided (shouldn't happen with updated train.py)
            self.episode_lengths.append(0)
            self.avg_lengths.append(0)
        
        self._update_graph_lines()
        
    def _update_graph_lines(self):
        """Update the graph plot lines with current data."""
        if not self.episodes:
            return
            
        self.line_reward.set_data(self.episodes, self.rewards)
        self.line_avg.set_data(self.episodes, self.avg_rewards)
        
        # Update Steps plot
        if self.episode_lengths:
            # Ensure lengths match episodes (in case of legacy data mismatch)
            min_len = min(len(self.episodes), len(self.episode_lengths))
            self.line_steps.set_data(self.episodes[:min_len], self.episode_lengths[:min_len])
            self.line_avg_steps.set_data(self.episodes[:min_len], self.avg_lengths[:min_len])
            
            # Auto-scale y-axis for steps
            if len(self.episode_lengths) > 1:
                max_steps = max(self.episode_lengths)
                self.ax_steps.set_ylim(0, max_steps * 1.1)
        
        # Auto-scale x-axis
        max_ep = max(self.episodes)
        self.ax_reward.set_xlim(0, max(50, max_ep + 10))
        self.ax_steps.set_xlim(0, max(50, max_ep + 10))
        
        # Update epsilon text
        if self.epsilons:
            self.epsilon_text.set_text(f"ε = {self.epsilons[-1]:.4f}")
        
        # Auto-scale y-axis for rewards
        if len(self.rewards) > 1:
            min_r = min(self.rewards)
            max_r = max(self.rewards)
            margin = max(3, (max_r - min_r) * 0.2)
            self.ax_reward.set_ylim(min_r - margin, max_r + margin)
            
    def _render_graph(self) -> np.ndarray:
        """Render the graph to a numpy array."""
        # Draw the figure to a canvas
        self.fig.canvas.draw()
        
        # Convert to numpy array
        buf = self.fig.canvas.buffer_rgba()
        graph_rgba = np.asarray(buf)
        graph_rgb = cv2.cvtColor(graph_rgba, cv2.COLOR_RGBA2RGB)
        
        # Resize to match game dimensions
        if graph_rgb.shape[:2] != (self.GRAPH_HEIGHT, self.GRAPH_WIDTH):
            graph_rgb = cv2.resize(graph_rgb, (self.GRAPH_WIDTH, self.GRAPH_HEIGHT))
            
        return graph_rgb
    
    def _get_all_frame_paths(self) -> List[str]:
        """
        Get all frame file paths from both legacy flat structure and batch subdirectories.
        Returns full paths sorted by frame number for continuous video export.
        """
        all_frames = []
        
        if not os.path.exists(self.frames_dir):
            return []
        
        for entry in os.listdir(self.frames_dir):
            entry_path = os.path.join(self.frames_dir, entry)
            
            # Check for legacy flat structure (frames directly in frames_dir)
            if entry.startswith("frame_") and (entry.endswith(".jpg") or entry.endswith(".png")):
                # Extract frame number from filename like "frame_000001.jpg"
                try:
                    frame_num = int(entry.split("_")[1].split(".")[0])
                    all_frames.append((frame_num, entry_path))
                except (IndexError, ValueError):
                    continue
            
            # Check batch subdirectories
            elif os.path.isdir(entry_path) and entry.startswith("batch_"):
                for frame_file in os.listdir(entry_path):
                    if frame_file.startswith("frame_") and (frame_file.endswith(".jpg") or frame_file.endswith(".png")):
                        try:
                            frame_num = int(frame_file.split("_")[1].split(".")[0])
                            all_frames.append((frame_num, os.path.join(entry_path, frame_file)))
                        except (IndexError, ValueError):
                            continue
        
        # Sort by frame number and return just the paths
        all_frames.sort(key=lambda x: x[0])
        return [path for _, path in all_frames]
        
    def export_video(
        self, 
        output_path: Optional[str] = None, 
        fps: int = 30,
        quality: str = "high",
        layout: str = "side-by-side",
        limit: Optional[int] = None
    ) -> str:
        """
        Export saved frames to an MP4 video file.
        
        Args:
            output_path: Output video path (default: checkpoints/training_recording.mp4)
            fps: Frames per second (default: 30)
            quality: Video quality - 'high', 'medium', or 'low' (affects file size)
            layout: Frame layout - 'side-by-side' (game|graph), 'game-only', or 'graph-only'
            limit: Only export the last N frames (useful for long sessions)
            
        Returns:
            Path to the exported video
        """
        if output_path is None:
            output_path = os.path.join(self.checkpoint_dir, "training_recording.mp4")
        
        # Ensure .mp4 extension
        if not output_path.lower().endswith('.mp4'):
            output_path = os.path.splitext(output_path)[0] + '.mp4'
            
        # Get list of all frame files from both legacy flat structure and batch subdirectories
        frame_paths = self._get_all_frame_paths()
        
        if not frame_paths:
            print("Recorder: No frames to export")
            return ""
            
        # Apply limit if specified
        if limit is not None and limit > 0:
            frame_paths = frame_paths[-limit:]
            print(f"Recorder: Limiting export to last {limit} frames")
            
        print(f"Recorder: Exporting {len(frame_paths)} frames to video...")
        print(f"  Layout: {layout}, Quality: {quality}, FPS: {fps}")
        
        # Read first frame to get dimensions and determine crop region
        first_frame = cv2.imread(frame_paths[0])
        full_height, full_width = first_frame.shape[:2]
        
        # Determine output dimensions based on layout
        if layout == "game-only":
            # Left half only (game)
            crop_region = (0, 0, full_width // 2, full_height)
            width, height = full_width // 2, full_height
            print(f"  Output: Game only ({width}x{height})")
        elif layout == "graph-only":
            # Right half only (graph)
            crop_region = (full_width // 2, 0, full_width, full_height)
            width, height = full_width // 2, full_height
            print(f"  Output: Graph only ({width}x{height})")
        else:  # side-by-side (default)
            crop_region = None
            width, height = full_width, full_height
            print(f"  Output: Side-by-side ({width}x{height})")
        
        # Scale based on quality
        scale_factors = {"high": 1.0, "medium": 0.75, "low": 0.5}
        scale = scale_factors.get(quality, 1.0)
        if scale != 1.0:
            width = int(width * scale)
            height = int(height * scale)
            print(f"  Scaled to: {width}x{height}")
        
        # Use imageio with ffmpeg for reliable MP4 output
        try:
            import imageio
            use_imageio = True
            print(f"  Using imageio with FFmpeg for MP4 output")
        except ImportError:
            use_imageio = False
            print("  imageio not available, falling back to OpenCV")
        
        final_output_path = os.path.splitext(output_path)[0] + '.mp4'
        
        if use_imageio:
            # Use imageio-ffmpeg for reliable MP4 encoding
            try:
                writer = imageio.get_writer(
                    final_output_path,
                    fps=fps,
                    codec='libx264',
                    quality=8,  # 0-10, higher = better quality
                    pixelformat='yuv420p',  # Compatibility with most players
                    macro_block_size=16
                )
                
                # Write all frames
                for i, frame_path in enumerate(frame_paths):
                    frame = cv2.imread(frame_path)
                    
                    if frame is None:
                        continue
                    
                    # Apply crop if needed
                    if crop_region is not None:
                        x1, y1, x2, y2 = crop_region
                        frame = frame[y1:y2, x1:x2]
                    
                    # Apply scale if needed
                    if scale != 1.0:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    
                    # Convert BGR (OpenCV) to RGB (imageio)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(frame_rgb)
                    
                    # Progress update
                    if (i + 1) % 500 == 0 or (i + 1) == len(frame_paths):
                        print(f"  Exported {i + 1}/{len(frame_paths)} frames...")
                
                writer.close()
                
            except Exception as e:
                print(f"  imageio export failed: {e}")
                print("  Falling back to OpenCV...")
                use_imageio = False
        
        if not use_imageio:
            # Fallback to OpenCV with MJPG/AVI
            final_output_path = os.path.splitext(output_path)[0] + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                print("Recorder: ERROR - Could not create video writer!")
                return ""
            
            for i, frame_path in enumerate(frame_paths):
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    continue
                
                if crop_region is not None:
                    x1, y1, x2, y2 = crop_region
                    frame = frame[y1:y2, x1:x2]
                
                if scale != 1.0:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                
                writer.write(frame)
                
                if (i + 1) % 500 == 0 or (i + 1) == len(frame_paths):
                    print(f"  Exported {i + 1}/{len(frame_paths)} frames...")
            
            writer.release()
        
        # Verify file was created and has content
        if os.path.exists(final_output_path):
            size_mb = os.path.getsize(final_output_path) / (1024 * 1024)
            if size_mb > 0.01:  # More than 10KB
                print(f"Recorder: Video exported successfully!")
                print(f"  Path: {final_output_path}")
                print(f"  Size: {size_mb:.1f} MB")
                print(f"  Duration: ~{len(frame_paths) / fps:.1f} seconds")
                return final_output_path
            else:
                print(f"Recorder: WARNING - Video file is too small ({size_mb:.3f} MB)")
                print("  The codec may not be working correctly.")
        
        return final_output_path
        
    def close(self):
        """Clean up resources."""
        # Ensure all buffered frames are written before closing
        self.flush_remaining()
        plt.close(self.fig)


def export_training_video(
    checkpoint_dir: str = "checkpoints",
    output_path: Optional[str] = None,
    fps: int = 30,
    quality: str = "high",
    layout: str = "side-by-side",
    limit: Optional[int] = None
) -> str:
    """
    Standalone function to export training video.
    
    Args:
        checkpoint_dir: Directory containing training frames
        output_path: Output video path
        fps: Frames per second
        quality: Video quality - 'high', 'medium', or 'low'
        layout: Frame layout - 'side-by-side', 'game-only', or 'graph-only'
        limit: Only export the last N frames
        
    Returns:
        Path to the exported video
    """
    recorder = TrainingRecorder(checkpoint_dir=checkpoint_dir)
    recorder.start()
    result = recorder.export_video(output_path, fps, quality, layout, limit)
    recorder.close()
    return result


if __name__ == "__main__":
    # Test the recorder
    print("Testing TrainingRecorder...")
    
    recorder = TrainingRecorder(checkpoint_dir="checkpoints", frame_skip=1)
    recorder.start()
    
    # Simulate some frames
    for step in range(100):
        # Create a dummy game frame
        game_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        game_frame[:, :, 1] = 50  # Dark green background
        
        # Add a "ball" that moves
        ball_x = int(320 + 300 * np.sin(step * 0.1))
        ball_y = int(240 + 200 * np.sin(step * 0.15))
        cv2.circle(game_frame, (ball_x, ball_y), 10, (255, 255, 255), -1)
        
        recorder.add_frame(game_frame)
        
        # Simulate episode end every 20 steps
        if step > 0 and step % 20 == 0:
            episode = step // 20
            reward = -21 + episode * 2 + np.random.randn() * 2
            avg_reward = -21 + episode * 1.5
            epsilon = 1.0 - episode * 0.1
            recorder.update_graph(episode, reward, avg_reward, epsilon)
    
    recorder.save()
    recorder.export_video(fps=10)
    recorder.close()
    
    print("Test complete!")
