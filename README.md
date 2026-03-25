# 🏓 Pong AI - Deep Q-Learning Agent

A self-learning AI that masters Atari Pong using Deep Q-Networks (DQN).

![Pong](https://upload.wikimedia.org/wikipedia/en/thumb/0/03/Pong.svg/220px-Pong.svg.png)

## 🚀 Quick Start

### Step 1: Install Dependencies

Open Command Prompt in this folder and run:

```bash
pip install -r requirements.txt
```

### Step 2: Install Atari ROMs

```bash
AutoROM --accept-license
```

### Step 3: Test Your Setup

```bash
python main.py test
```

You should see all checkmarks (✓) if everything is configured correctly.

### Step 4: Train the Agent

```bash
python main.py train
```

Training will take several hours on CPU. Progress is saved every 50 episodes.

### Step 5: Watch It Play!

```bash
python main.py play
```

---

## 📁 Project Structure

```
pong-ai/
├── main.py           # Entry point (train/play/test commands)
├── train.py          # Training loop with progress tracking
├── play.py           # Watch trained agent play
├── agent.py          # DQN agent (epsilon-greedy, learning)
├── model.py          # Convolutional neural network
├── replay_buffer.py  # Experience replay storage
├── environment.py    # Atari Pong wrapper (preprocessing)
├── requirements.txt  # Python dependencies
└── checkpoints/      # Saved models (created during training)
```

---

## 🧠 How It Works

### 1. State Representation
- Raw game frames → Grayscale → Resize to 84×84
- Stack 4 consecutive frames = sense of motion/velocity
- Input to network: **4 × 84 × 84 tensor**

### 2. Neural Network (CNN)
```
Input [4×84×84] 
  → Conv(32, 8×8, stride=4) → ReLU
  → Conv(64, 4×4, stride=2) → ReLU  
  → Conv(64, 3×3, stride=1) → ReLU
  → Flatten → Dense(512) → ReLU
  → Dense(3) [Q-values for: NOOP, UP, DOWN]
```

### 3. Learning Algorithm (DQN)
1. **Epsilon-Greedy Action Selection**: Start with random actions (explore), gradually shift to best actions (exploit)
2. **Experience Replay**: Store (state, action, reward, next_state) tuples, sample random batches to break correlation
3. **Target Network**: Separate network updated periodically for stable Q-value targets
4. **Bellman Update**: `Q(s,a) ← r + γ * max Q_target(s', a')`

---

## 📊 Training Tips

| Parameter | Default | Notes |
|-----------|---------|-------|
| Episodes | 500 | More = better. Try 1000-2000 for strong agent |
| Epsilon decay | 100,000 steps | How fast exploration decreases |
| Buffer size | 100,000 | More memory = smoother learning |
| Batch size | 32 | Increase if you have more RAM |

### Expected Training Progress
- **Episodes 1-100**: Random behavior, learning begins
- **Episodes 100-300**: Starts tracking the ball
- **Episodes 300-500**: Develops defensive play
- **Episodes 500+**: Offensive strategies emerge

---

## 🎮 Commands Reference

```bash
# Train for 1000 episodes
python main.py train --episodes 1000

# Watch specific checkpoint play
python main.py play --checkpoint checkpoints/pong_dqn_ep500.pt

# Watch 10 games
python main.py play --episodes 10

# Record high-quality evaluation video (5 episodes, game-only layout)
python main.py record-eval
```

---

## 🔧 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "ROM not found" error
```bash
AutoROM --accept-license
```

### Training too slow
- This is expected on CPU (Intel GPU)
- Each episode takes ~30-60 seconds
- 500 episodes ≈ 4-8 hours
- Training overnight recommended!

---

## 📈 Results

After training, you'll find:
- `checkpoints/pong_dqn_best.pt` - Best performing model
- `checkpoints/pong_dqn_final.pt` - Final model
- `checkpoints/training_progress.png` - Learning curves

---

## 📚 References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - Original DQN paper
- [OpenAI Gymnasium](https://gymnasium.farama.org/) - Environment framework
- [PyTorch](https://pytorch.org/) - Deep learning library

---

Made with ❤️ for learning reinforcement learning!
