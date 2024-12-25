import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
import os
from main import Bird, Pipe, WIDTH, HEIGHT, PIPE_WIDTH, PIPE_GAP

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Increased network size
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)   # Output: [don't jump, jump]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class FlappyBirdAI:
    def __init__(self, model_path="best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        
        # Try to load existing model
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"Successfully loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)  # Reduced learning rate
        self.memory = deque(maxlen=100000)
        
        self.batch_size = 64  # Increased batch size
        self.gamma = 0.99
        self.epsilon = 0.1  # Reduced starting epsilon for loaded models
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.target_update = 5
        self.training_step = 0
        self.frame_stack = []  # For frame skipping
        self.skip_frames = 4   # Number of frames to skip

    def get_state(self, bird, pipes):
        # Find the next pipe
        next_pipe = None
        for pipe in pipes:
            if pipe.x + PIPE_WIDTH > bird.x:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            return torch.zeros(4).to(self.device)

        # Normalize state values
        state = [
            (bird.y - HEIGHT/2) / (HEIGHT/2),  # Centered and normalized bird height
            bird.velocity / 10.0,              # Normalized bird velocity
            (next_pipe.x - bird.x) / WIDTH,    # Normalized distance to next pipe
            (next_pipe.gap_y - bird.y) / HEIGHT # Normalized distance to pipe gap
        ]
        return torch.FloatTensor(state).to(self.device)

    def act(self, state, training=True):
        # Use epsilon-greedy only during training
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Double DQN implementation
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_actions = self.model(next_states).max(1)[1].detach()
        next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename):
        print(f"Saving model to {filename}")
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        if os.path.exists(filename):
            print(f"Loading model from {filename}")
            self.model.load_state_dict(torch.load(filename, map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict())
            self.epsilon = self.epsilon_min  # Set epsilon to minimum for loaded models
        else:
            print(f"No model found at {filename}")