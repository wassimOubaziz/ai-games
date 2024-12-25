import pygame
from main import PongGame
from learning import PongAI
import numpy as np
import argparse

def train_ai(episodes=1000, model_path="pong_model.pth", training_speed=3):
    game = PongGame(ai_opponent=True)
    ai = PongAI(model_path=model_path)
    
    # Increase game speed for faster training
    game.ball.speed_x *= training_speed
    game.ball.speed_y *= training_speed
    game.player.speed *= training_speed
    game.opponent.speed *= training_speed
    
    best_score = float('-inf')
    scores = []
    
    for episode in range(episodes):
        state = game.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Simple AI for the first player (training opponent)
            if game.ball.y > game.player.get_center() + 10:
                game.player.move(up=False)
            elif game.ball.y < game.player.get_center() - 10:
                game.player.move(up=True)
            
            # Get AI action
            action = ai.act(state)
            
            # Take action and get reward
            next_state, reward, done = game.step(action)
            
            # Remember the experience
            ai.remember(state, action, reward, next_state, done)
            
            # Train the model
            ai.replay()
            
            state = next_state
            episode_reward += reward
            
            # Render the game
            game.render()
            pygame.time.delay(max(1, int(16/training_speed)))  # Adjust delay based on training speed
        
        scores.append(episode_reward)
        avg_score = np.mean(scores[-100:])
        
        print(f"Episode: {episode + 1}, Score: {episode_reward}, Avg Score: {avg_score:.2f}, Epsilon: {ai.epsilon:.2f}")
        
        # Save if we have a new best model
        if avg_score > best_score and episode > 100:
            best_score = avg_score
            ai.save(model_path)
            print(f"New best model saved with average score: {avg_score:.2f}")

def play_against_ai(model_path="pong_model.pth"):
    game = PongGame(ai_opponent=True)
    ai = PongAI(model_path=model_path, fine_tune=False)
    
    running = True
    while running:
        running, _ = game.play_step(ai_agent=ai)
    
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pong AI Training and Playing')
    parser.add_argument('--mode', type=str, choices=['train', 'play'], default='play',
                      help='Mode to run the program in (train or play)')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of episodes to train (only used in train mode)')
    parser.add_argument('--model', type=str, default='pong_model.pth',
                      help='Path to the model file')
    parser.add_argument('--speed', type=int, default=3,
                      help='Training speed multiplier')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ai(episodes=args.episodes, model_path=args.model, training_speed=args.speed)
    else:
        play_against_ai(model_path=args.model)
