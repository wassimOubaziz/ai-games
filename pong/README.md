# AI Pong 🏓

A modern implementation of the classic Pong game with AI learning capabilities. Play against the computer or watch the AI train itself to become a Pong master!

## Game Features

- **Smooth Physics**: Realistic ball and paddle physics
- **Dynamic Angles**: Ball direction changes based on hit position
- **Score Tracking**: Keep track of both players' scores
- **Speed Control**: Adjustable game speed for training

## AI Features

- **Deep Q-Learning**: Advanced AI learning system
- **Experience Priority**: Learns more from important game moments
- **Continuous Learning**: Can be fine-tuned from previous training
- **Real-time Training**: Watch the AI improve as it plays

## How to Play

### Playing Yourself
```bash
# Start a regular game where you play against a basic AI
python main.py
```

**Controls & Gameplay:**
- Use UP ARROW to move your paddle up
- Use DOWN ARROW to move your paddle down
- You control the left paddle
- Hit the ball past the AI's paddle to score
- First to reach the winning score wins!
- Ball speed increases with each hit
- Ball angle changes based on where it hits the paddle

**Tips for Playing:**
- Try to predict the ball's trajectory
- Hit with the paddle's edges for sharper angles
- Stay near the center when possible
- Move early to get in position
- Use the paddle edges for more challenging returns

### Play Against AI
```bash
python train.py --mode watch
```
Optional arguments:
- `--speed`: Game speed multiplier (default: 1)
- `--model`: Path to model file

### Train the AI
```bash
python train.py --mode train --episodes 1000 --speed 5
```
Arguments:
- `--episodes`: Number of training episodes
- `--speed`: Training speed multiplier
- `--plot`: Show training progress plot
- `--model`: Model save/load path

## Controls

- **Player Controls**:
  - UP ARROW: Move paddle up
  - DOWN ARROW: Move paddle down
  - ESC: Quit game

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Technical Details

### State Space
The AI observes:
- Ball position and velocity
- Both paddles' positions
- Relative distances and angles

### Action Space
Three possible actions:
- Stay in position
- Move up
- Move down

### Reward System
- +1 for scoring a point
- -1 for losing a point
- Small rewards for good positioning
- Small penalties for poor positioning

## Training Tips

1. **Initial Training**:
   - Start with higher speed (5-10x)
   - Train for at least 500 episodes
   - Watch for consistent scoring

2. **Fine-Tuning**:
   - Load previous model
   - Use lower learning rate
   - Train additional episodes

## Performance Metrics

Typical AI progression:
- Basic returns: ~100 episodes
- Consistent rallies: ~300 episodes
- Strategic play: ~500 episodes
- Expert level: ~1000 episodes

## Known Strategies

The AI typically develops these strategies:
- Center position waiting
- Angle manipulation
- Aggressive positioning
- Recovery movements
