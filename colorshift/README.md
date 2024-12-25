# ColorShift 🌈

A beautiful platformer game where you master the art of color-shifting to navigate through an ever-changing world. Features stunning visual effects, dynamic platforms, and an AI that learns to master both movement and color-changing strategies.

## Game Features

- **Color-Shifting Mechanics**:
  - Switch between 4 different colors (Red, Blue, Green, Yellow)
  - Only interact with platforms of matching color
  - Beautiful particle effects during color changes
  - Glowing platforms and player

- **Dynamic Gameplay**:
  - Endless procedurally generated levels
  - Moving platforms
  - Physics-based movement
  - Particle effects
  - Score tracking

- **AI Learning**:
  - Deep Q-Learning with experience replay
  - Priority memory for important experiences
  - Double DQN architecture
  - Dynamic exploration strategies
  - Performance visualization

## How to Play

### Playing Yourself
```bash
python main.py
```

**Controls**:
- LEFT/RIGHT ARROW: Move left/right
- SPACE: Jump (press twice for double jump!)
- 1-4 Keys: Change colors
  - 1: Red
  - 2: Blue
  - 3: Green
  - 4: Yellow

**Tips**:
- Use double jump to reach higher platforms
- Save your second jump for emergencies
- Match your color to the platforms you want to land on
- Plan your color changes ahead of time
- Use moving platforms to reach higher areas
- Watch for patterns in platform colors
- Time your jumps carefully

### Train AI
```bash
# Train with default settings
python train.py --mode train --episodes 1000

# Train with visualization
python train.py --mode train --episodes 1000 --plot

# Train with custom speed
python train.py --mode train --episodes 1000 --speed 5
```

### Watch AI Play
```bash
# Watch at normal speed
python train.py --mode watch

# Watch at custom speed
python train.py --mode watch --speed 2
```

## Requirements

Install required packages:
```bash
pip install -r requirements.txt
```

## Technical Details

### State Space
The AI observes:
- Player position and velocity
- Current color
- Nearby platform positions, sizes, and colors
- Platform movement patterns

### Action Space
Eight possible actions:
- No action
- Move left
- Move right
- Jump
- Change to Red
- Change to Blue
- Change to Green
- Change to Yellow

### Reward System
- Height-based rewards
- Successful platform landings
- Efficient color changes
- Survival time
- Penalties for falls

## AI Training Tips

1. **Training Phases**:
   - Early Phase (0-200 episodes): Basic movement
   - Mid Phase (200-500 episodes): Color coordination
   - Late Phase (500+ episodes): Advanced strategies

2. **Speed Selection**:
   - Training: 5x-20x recommended
   - Watching: 1x-3x for best visualization

3. **Training Strategy**:
   - Start with shorter episodes
   - Gradually increase difficulty
   - Save best performing models
   - Fine-tune from saved states

## Performance Expectations

The AI typically progresses through these stages:
1. **Basic Movement** (~100 episodes)
   - Learning to jump
   - Simple platform targeting

2. **Color Management** (~300 episodes)
   - Basic color matching
   - Simple platform sequences

3. **Advanced Strategies** (~500 episodes)
   - Efficient color switching
   - Complex platform sequences
   - Using moving platforms

4. **Expert Play** (~1000 episodes)
   - Optimal path finding
   - Perfect color timing
   - High score achievement
