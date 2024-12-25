import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

class DQN(nn.Module):
    def __init__(self, input_size=6, hidden_size=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)  # 3 actions: no move, up, down
        )
        
    def forward(self, x):
        return self.network(x)

class PongAI:
    def __init__(self, model_path="pong_model.pth", fine_tune=True):
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
        
        # Load model if it exists
        if os.path.exists(model_path):
            try:
                self.load(model_path, fine_tune=fine_tune)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with a fresh model")

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
                    param_group['lr'] = 0.0001
                
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
