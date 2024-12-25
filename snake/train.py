import pygame
from main import SnakeGame
from learning import SnakeAI
import numpy as np
import argparse
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

def plot_training(scores, mean_scores):
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Average Score')
    plt.ylim(ymin=0)
    plt.legend()
    plt.pause(0.1)

def train_ai(episodes=1000, model_path="snake_model.pth", plot_progress=True, speed=5):
    game = SnakeGame(enable_render=True, speed=speed)
    ai = SnakeAI(model_path=model_path)
    
    scores = []
    mean_scores = []
    total_score = 0
    best_score = 0
    
    if plot_progress:
        plt.ion()
    
    for episode in range(episodes):
        state = game.reset()
        done = False
        episode_score = 0
        
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Get AI action
            action = ai.act(state)
            
            # Take action
            next_state, reward, done = game.step(action)
            
            # Remember the experience
            ai.remember(state, action, reward, next_state, done)
            
            # Train the model
            if len(ai.memory) > ai.batch_size:
                loss = ai.replay()
            
            state = next_state
            episode_score += reward
            
            # Render game
            game.render()
        
        # Update scores
        scores.append(episode_score)
        total_score += game.score
        mean_score = total_score / (episode + 1)
        mean_scores.append(mean_score)
        
        if game.score > best_score:
            best_score = game.score
            ai.save(model_path)
        
        print(f'Game {episode + 1} Score: {game.score} Average: {mean_score:.2f} ' \
              f'Best: {best_score} Epsilon: {ai.epsilon:.2f}')
        
        if plot_progress and episode % 5 == 0:
            plot_training(scores, mean_scores)
    
    if plot_progress:
        plt.ioff()
        plt.show()

def watch_ai(model_path="snake_model.pth", speed=1):
    game = SnakeGame(speed=speed)
    ai = SnakeAI(model_path=model_path, fine_tune=False)
    
    running = True
    while running:
        state = game.reset()
        done = False
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            action = ai.act(state, training=False)
            state, _, done = game.step(action)
            game.render()
            
            if done:
                print(f"Game Over! Score: {game.score}")
                time.sleep(1)  # Pause briefly between games
    
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Snake AI Training and Demo')
    parser.add_argument('--mode', type=str, choices=['train', 'watch'], default='watch',
                      help='Mode to run the program in (train or watch)')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of episodes to train (only used in train mode)')
    parser.add_argument('--model', type=str, default='snake_model.pth',
                      help='Path to the model file')
    parser.add_argument('--plot', action='store_true',
                      help='Plot training progress (only used in train mode)')
    parser.add_argument('--speed', type=int, default=5,
                      help='Game speed multiplier (default: 5)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ai(episodes=args.episodes, model_path=args.model, 
                plot_progress=args.plot, speed=args.speed)
    else:
        watch_ai(model_path=args.model, speed=args.speed)
