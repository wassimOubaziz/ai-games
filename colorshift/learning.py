import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

class DQN(nn.Module):
    def __init__(self, input_size=17, hidden_size=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 8)  # 8 actions: nothing, left, right, jump, 4 colors
        )
        
    def forward(self, x):
        return self.network(x)

class ColorShiftAI:
    def __init__(self, model_path="model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=100000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.steps = 0
        
        if os.path.exists(model_path):
            self.load_model(model_path)
            print("Loaded existing model")
            self.epsilon = max(self.epsilon_min, self.epsilon)  # Start with some exploration
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            # During exploration, encourage more jumping and color changes
            if random.random() < 0.4:
                return random.choice([3, 4, 5, 6, 7])  # Jump or color changes
            return random.randint(0, 7)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([s[0] for s in batch]).to(self.device)
        actions = torch.LongTensor([s[1] for s in batch]).to(self.device)
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device)
        next_states = torch.FloatTensor([s[3] for s in batch]).to(self.device)
        dones = torch.FloatTensor([s[4] for s in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.model(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps = checkpoint.get('steps', 0)
        print(f"Model loaded from {path}")
