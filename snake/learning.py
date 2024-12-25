import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

class DQN(nn.Module):
    def __init__(self, input_size=11, hidden_size=256):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 3)  # 3 actions: [straight, right, left]
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)

class SnakeAI:
    def __init__(self, model_path="snake_model.pth", fine_tune=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Initialize training parameters
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.training_step = 0
        
        # Priority replay parameters
        self.priority_memory = deque(maxlen=10000)
        self.priority_prob = 0.4
        
        # Load model if it exists
        if os.path.exists(model_path):
            try:
                self.load(model_path, fine_tune=fine_tune)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with a fresh model")

    def act(self, state, training=True):
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        # Add to regular memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Add to priority memory if reward is significant
        if abs(reward) > 0.5:
            self.priority_memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Mix regular and priority samples
        priority_size = int(self.batch_size * self.priority_prob)
        regular_size = self.batch_size - priority_size
        
        # Get samples
        regular_batch = random.sample(self.memory, regular_size)
        priority_batch = random.sample(self.priority_memory, 
                                     min(priority_size, len(self.priority_memory)))
        batch = regular_batch + priority_batch
        
        # Prepare batch
        states = torch.FloatTensor([x[0] for x in batch]).to(self.device)
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1]
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards + (1 - dones) * self.gamma * next_q.squeeze()
        
        # Compute loss and update
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def save(self, filename):
        print(f"Saving model to {filename}")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'memory': list(self.priority_memory)  # Save priority experiences
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
                    param_group['lr'] = 0.0001
                
                # Load previous training state
                self.training_step = checkpoint.get('training_step', 0)
                
                # Load saved experiences if available
                if 'memory' in checkpoint:
                    saved_memory = checkpoint['memory']
                    self.priority_memory.extend(saved_memory)
                    print(f"Loaded {len(saved_memory)} priority experiences")
            else:
                # For inference only
                self.epsilon = self.epsilon_min
                print("Model loaded for inference only")
        else:
            print(f"No model found at {filename}, starting with a new model")
