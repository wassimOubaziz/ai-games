# Neural Snake 🐍

A beautiful, modern implementation of the classic Snake game with AI learning capabilities. Features stunning visual effects and deep learning AI that masters the game through reinforcement learning.

## Game Features

- **Modern Graphics**:
  - Glowing food effects
  - Snake gradient coloring
  - Grid lighting effects
  - Smooth animations

- **Gameplay Elements**:
  - Progressive difficulty
  - Score tracking
  - High score system
  - Speed control

## AI Features

- **Advanced Learning**:
  - Deep Q-Network (DQN)
  - Priority Experience Replay
  - Double DQN implementation
  - Adaptive exploration rates

- **Training Visualization**:
  - Real-time score plotting
  - Performance metrics
  - Training progress display
  - Speed control

## How to Play

### Playing Yourself
```bash
# Start the game in player mode
python main.py
```

**Controls & Gameplay:**
- Use LEFT ARROW to turn snake left
- Use RIGHT ARROW to turn snake right
- Collect the glowing food to grow
- Avoid hitting walls
- Don't run into your own tail
- Score increases with each food eaten
- Game gets faster as you grow longer
- Try to beat the high score!

**Tips for Playing:**
- Plan your path several moves ahead
- Use the grid lines to guide your movement
- Leave yourself room to maneuver
- Don't get trapped against walls
- Try to keep a clear path to food
- Use the whole game area efficiently

### Train AI
```bash
# Train with default settings
python train.py --mode train --episodes 1000

# Train with visualization
python train.py --mode train --episodes 1000 --plot

# Train with custom speed
python train.py --mode train --episodes 1000 --speed 10
```

### Watch AI Play
```bash
# Watch at normal speed
python train.py --mode watch

# Watch at custom speed
python train.py --mode watch --speed 2
```

## Controls

- **Manual Mode**:
  - LEFT ARROW: Turn left
  - RIGHT ARROW: Turn right
  - ESC: Quit game

## Requirements

Install required packages:
```bash
pip install -r requirements.txt
```

## Technical Details

### State Space
The AI observes:
- Snake head position
- Food location
- Danger in each direction
- Current direction
- Body position

### Action Space
Three possible actions:
- Continue straight
- Turn right
- Turn left

### Reward System
- +10 for eating food
- -10 for collision
- Small rewards for moving towards food
- Small penalties for moving away

## Training Tips

1. **Speed Selection**:
   - Training: 5x-20x recommended
   - Watching: 1x-3x for best visualization

2. **Training Strategy**:
   - Start with shorter episodes
   - Gradually increase difficulty
   - Save best performing models
   - Fine-tune from saved states

## Performance Expectations

The AI typically progresses through these stages:
1. **Basic Movement** (~50 episodes)
   - Learns to avoid walls
   - Basic food seeking

2. **Intermediate Play** (~200 episodes)
   - Efficient pathfinding
   - Avoiding self-collisions

3. **Advanced Strategies** (~500 episodes)
   - Optimal path planning
   - Space utilization
   - Long-term survival

4. **Expert Play** (~1000 episodes)
   - Consistent high scores
   - Efficient space usage
   - Advanced survival tactics

## Customization

You can modify various parameters in the code:
- Grid size
- Game speed
- Learning rate
- Reward values
- Network architecture
