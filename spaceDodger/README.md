# Space Dodger 🚀

A thrilling space-themed game where you pilot a spaceship through an asteroid field. Features both manual play and AI learning capabilities.

## Game Features

- **Dynamic Obstacles**: Asteroids with varying sizes and speeds
- **Physics-Based Movement**: Smooth spaceship controls with momentum
- **Progressive Difficulty**: Game becomes harder as your score increases
- **Beautiful Graphics**: Space-themed visuals with particle effects

## AI Features

- **Deep Q-Learning**: Advanced AI that learns optimal dodge patterns
- **Experience Replay**: Efficient learning from past experiences
- **Fine-Tuning**: Can continue training from previously saved models
- **Adjustable Parameters**: Customize learning rate, exploration, etc.

## How to Play

### Playing Yourself
```bash
# Start the game in player mode
python main.py
```

**Controls & Gameplay:**
- Press and hold UP ARROW to thrust upward
- Release UP ARROW to fall
- Avoid incoming asteroids
- Your score increases the longer you survive
- Game gets progressively harder
- Try to beat your high score!

**Tips for Playing:**
- Use short, controlled bursts of thrust
- Stay near the middle of the screen when possible
- Watch for patterns in asteroid movement
- Don't hold thrust too long - you might overshoot
- Plan your moves ahead of time

### Train AI
```bash
python train.py --mode train --episodes 1000
```
Optional arguments:
- `--episodes`: Number of training episodes
- `--model`: Path to save/load model
- `--speed`: Training speed multiplier

### Watch AI Play
```bash
python train.py --mode watch
```

## Controls

- **Manual Mode**:
  - UP ARROW: Thrust
  - ESC: Quit game

- **Training Mode**:
  - ESC: Stop training
  - SPACE: Pause/Resume

## Requirements

Install required packages:
```bash
pip install -r requirements.txt
```

## Technical Details

### State Space
The AI observes:
- Ship position and velocity
- Nearest obstacles' positions and speeds
- Current difficulty level

### Action Space
- No thrust
- Apply thrust

### Reward System
- Positive reward for surviving
- Negative reward for collisions
- Bonus rewards for close calls
- Progressive rewards for higher scores

## Tips

1. **For Playing**:
   - Use short bursts of thrust for precise control
   - Plan your movement ahead of time
   - Watch out for asteroid patterns

2. **For Training**:
   - Start with higher exploration rate
   - Gradually increase difficulty
   - Save best models for future fine-tuning

## Performance

The AI typically achieves:
- Basic competence after ~100 episodes
- Good performance after ~500 episodes
- Expert play after ~1000 episodes
