# AI Game Collection

A collection of classic games reimagined with modern graphics and AI learning capabilities. Each game features both playable modes and trainable AI that learns through deep reinforcement learning.

## Games Included

### 1. Space Dodger 🚀
A space-themed game where you navigate a spaceship through obstacles. Features:
- Dynamic obstacle patterns
- Physics-based movement
- AI learns optimal dodge patterns

### 2. Pong 🏓
The classic Pong game with a modern twist. Features:
- Smooth paddle physics
- Dynamic ball angles
- AI learns strategic positioning

### 3. Snake 🐍
A beautiful modern version of the classic Snake game. Features:
- Glowing effects and modern graphics
- Gradient snake coloring
- AI learns optimal path finding

## Common Features

All games include:
- Modern, beautiful graphics
- Deep Q-Learning AI implementation
- Real-time training visualization
- Save/Load model functionality
- Adjustable training speeds
- Both play and watch modes

## Requirements

Each game has its own requirements.txt file, but generally needs:
```bash
pygame
torch
numpy
matplotlib
```

## Quick Start

1. Clone the repository:
```bash
git clone [repository-url]
cd ai-games
```

2. Navigate to any game directory:
```bash
cd [game-name]  # spaceDodger, pong, or snake
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Play the Games:

**Space Dodger**:
```bash
cd spaceDodger
python main.py  # Use UP ARROW to thrust and avoid asteroids
```

**Pong**:
```bash
cd pong
python main.py  # Use UP/DOWN ARROWS to move paddle
```

**Snake**:
```bash
cd snake
python main.py  # Use LEFT/RIGHT ARROWS to turn snake
```

5. Train or Watch AI:
```bash
python train.py --mode train  # To train the AI
# or
python train.py --mode watch  # To watch AI play
```

## Directory Structure
```
ai-games/
├── spaceDodger/
│   ├── main.py
│   ├── learning.py
│   ├── train.py
│   └── requirements.txt
├── pong/
│   ├── main.py
│   ├── learning.py
│   ├── train.py
│   └── requirements.txt
└── snake/
    ├── main.py
    ├── learning.py
    ├── train.py
    └── requirements.txt
```

For detailed information about each game, please refer to their individual README files in their respective directories.
