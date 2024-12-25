import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
import os
from main import Spaceship, Asteroid, WIDTH, HEIGHT, SHIP_SIZE, ASTEROID_SIZE

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Increased network size and added dropout for better generalization
        self.fc1 = nn.Linear(6, 256)  # Added one more input for difficulty
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)   # Output: [don't thrust, thrust]
        
        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class SpaceDodgerAI:
    def __init__(self, model_path="best_model.pth", fine_tune=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025)
        self.memory = deque(maxlen=100000)
        
        # Initialize training parameters
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9997
        self.target_update = 10
        self.training_step = 0
        
        # Experience replay priorities
        self.priority_weight = 0.6
        self.recent_memories = deque(maxlen=1000)
        
        # Load model if it exists
        if os.path.exists(model_path):
            try:
                self.load(model_path, fine_tune=fine_tune)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with a fresh model")

    def get_state(self, ship, asteroids):
        # Find the next two asteroids
        next_asteroids = []
        for asteroid in sorted(asteroids, key=lambda x: x.x):
            if asteroid.x + ASTEROID_SIZE > ship.x and len(next_asteroids) < 2:
                next_asteroids.append(asteroid)
        
        while len(next_asteroids) < 2:
            # Add dummy asteroids if we don't have enough
            dummy = Asteroid(WIDTH * 2)  # Far away asteroid
            next_asteroids.append(dummy)

        # Enhanced state representation
        state = [
            (ship.y - HEIGHT/2) / (HEIGHT/2),  # Normalized ship position (-1 to 1)
            ship.velocity / 10.0,              # Normalized velocity
            (next_asteroids[0].x - ship.x) / WIDTH,  # Distance to first asteroid
            (next_asteroids[0].y - ship.y) / HEIGHT, # Vertical distance to first asteroid
            (next_asteroids[1].x - ship.x) / WIDTH,  # Distance to second asteroid
            next_asteroids[0].speed / 10.0          # Speed of next asteroid
        ]
        return torch.FloatTensor(state).to(self.device)

    def act(self, state, training=True):
        # Epsilon-greedy with temperature scaling
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        
        with torch.no_grad():
            q_values = self.model(state)
            # Add temperature scaling for more decisive actions
            temperature = 1.0
            q_values = q_values / temperature
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        # Add experience to both regular and priority queues
        self.memory.append((state, action, reward, next_state, done))
        if reward != 0:  # Prioritize experiences with non-zero rewards
            self.recent_memories.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Mix regular and prioritized experiences
        regular_batch_size = self.batch_size // 2
        priority_batch_size = self.batch_size - regular_batch_size
        
        regular_batch = random.sample(self.memory, regular_batch_size)
        priority_batch = random.sample(self.recent_memories, 
                                     min(priority_batch_size, len(self.recent_memories)))
        
        batch = regular_batch + priority_batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Double DQN with TD-error clipping
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1]
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Huber loss for robustness
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename):
        print(f"Saving model to {filename}")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'memory': list(self.memory)[-1000:],  # Save last 1000 experiences
        }
        torch.save(checkpoint, filename)

    def load(self, filename, fine_tune=True):
        if os.path.exists(filename):
            print(f"Loading model from {filename}")
            checkpoint = torch.load(filename, map_location=self.device)
            
            # Load model and optimizer states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if fine_tune:
                # For fine-tuning, set a higher epsilon to allow some exploration
                self.epsilon = max(0.1, checkpoint.get('epsilon', 0.1))
                # Adjust learning rate for fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0001  # Lower learning rate for fine-tuning
                
                # Load previous training state
                self.training_step = checkpoint.get('training_step', 0)
                
                # Load saved experiences if available
                if 'memory' in checkpoint:
                    saved_memory = checkpoint['memory']
                    self.memory.extend(saved_memory)
                    print(f"Loaded {len(saved_memory)} previous experiences")
            else:
                # For inference only
                self.epsilon = self.epsilon_min
                print("Model loaded for inference only")
        else:
            print(f"No model found at {filename}, starting with a new model")
