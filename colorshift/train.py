import pygame
from main import ColorShiftGame
from learning import ColorShiftAI
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def plot_training(scores, mean_scores, losses):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Plot scores
    ax1.set_title('Training Progress')
    ax1.set_xlabel('Games')
    ax1.set_ylabel('Score')
    ax1.plot(scores, label='Score', alpha=0.4)
    ax1.plot(mean_scores, label='Average Score')
    ax1.legend()
    
    # Plot losses
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss')
    ax2.plot(losses, label='Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.pause(0.1)

def train_ai(episodes=1000, model_path="model.pth", plot_progress=True, speed=1):
    game = ColorShiftGame(enable_render=True)
    ai = ColorShiftAI(model_path=model_path)
    
    scores = []
    mean_scores = []
    losses = []
    total_score = 0
    best_score = float('-inf')
    
    if plot_progress:
        plt.ion()
    
    try:
        for episode in range(episodes):
            state = game.reset()
            episode_score = 0
            episode_losses = []
            steps = 0
            done = False
            
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
                    episode_losses.append(loss)
                
                state = next_state
                episode_score += reward
                steps += 1
                
                # Render game
                game.render()
                
                # Control speed
                if speed > 1:
                    pygame.time.delay(max(1, int(16/speed)))
                
                # Prevent infinite episodes
                if steps > 1000:
                    done = True
            
            # Update scores
            scores.append(episode_score)
            total_score += episode_score
            mean_score = total_score / (episode + 1)
            mean_scores.append(mean_score)
            
            # Update losses
            if episode_losses:
                avg_loss = sum(episode_losses) / len(episode_losses)
                losses.append(avg_loss)
            
            if episode_score > best_score:
                best_score = episode_score
                ai.save_model(model_path)
                print(f"New best score! Model saved")
            
            print(f'Game {episode + 1}/{episodes} Score: {episode_score:.2f} ' \
                  f'Average: {mean_score:.2f} Best: {best_score:.2f} ' \
                  f'Epsilon: {ai.epsilon:.3f} Steps: {steps}')
            
            if plot_progress and (episode + 1) % 5 == 0:
                plot_training(scores, mean_scores, losses)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        if plot_progress:
            plt.ioff()
            plt.show()
        
        # Save final model
        ai.save_model(model_path)
        print("Training finished. Final model saved.")

def watch_ai(model_path="model.pth", speed=1):
    game = ColorShiftGame()
    ai = ColorShiftAI(model_path=model_path)
    
    try:
        while True:
            state = game.reset()
            episode_score = 0
            done = False
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                action = ai.act(state, training=False)
                state, reward, done = game.step(action)
                episode_score += reward
                game.render()
                
                if speed > 1:
                    pygame.time.delay(max(1, int(16/speed)))
            
            print(f"Game Over! Score: {episode_score:.2f}")
            time.sleep(1)  # Pause briefly between games
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ColorShift AI Training and Demo')
    parser.add_argument('--mode', type=str, choices=['train', 'watch'], default='watch',
                      help='Mode to run the program in (train or watch)')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of episodes to train (only used in train mode)')
    parser.add_argument('--model', type=str, default='model.pth',
                      help='Path to the model file')
    parser.add_argument('--plot', action='store_true',
                      help='Plot training progress (only used in train mode)')
    parser.add_argument('--speed', type=int, default=1,
                      help='Game speed multiplier (default: 1)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ai(episodes=args.episodes, model_path=args.model, 
                plot_progress=args.plot, speed=args.speed)
    else:
        watch_ai(model_path=args.model, speed=args.speed)
