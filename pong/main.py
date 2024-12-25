import pygame
import sys
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WIDTH = 800
HEIGHT = 600
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 90
BALL_SIZE = 15
PADDLE_SPEED = 5
BALL_SPEED = 7
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Paddle:
    def __init__(self, x, is_ai=False):
        self.x = x
        self.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.rect = pygame.Rect(x, self.y, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.speed = PADDLE_SPEED
        self.is_ai = is_ai
        self.score = 0

    def move(self, up=True):
        if up and self.y > 0:
            self.y -= self.speed
        elif not up and self.y < HEIGHT - PADDLE_HEIGHT:
            self.y += self.speed
        self.rect.y = self.y

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)

    def get_center(self):
        return self.y + PADDLE_HEIGHT // 2

class Ball:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.rect = pygame.Rect(self.x, self.y, BALL_SIZE, BALL_SIZE)
        self.speed_x = BALL_SPEED if random.random() > 0.5 else -BALL_SPEED
        self.speed_y = random.uniform(-BALL_SPEED, BALL_SPEED)

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y
        
        # Wall collision (top/bottom)
        if self.y <= 0 or self.y >= HEIGHT - BALL_SIZE:
            self.speed_y *= -1
            
        self.rect.x = self.x
        self.rect.y = self.y

    def paddle_collision(self, paddle):
        if self.rect.colliderect(paddle.rect):
            # Adjust angle based on where the ball hits the paddle
            relative_intersect_y = paddle.get_center() - self.y
            normalized_intersect = relative_intersect_y / (PADDLE_HEIGHT / 2)
            bounce_angle = normalized_intersect * np.pi / 4  # Max 45 degree angle
            
            speed = np.sqrt(self.speed_x**2 + self.speed_y**2)
            
            # Ball is moving right
            if self.speed_x > 0:
                self.speed_x = -speed * np.cos(bounce_angle)
                self.speed_y = -speed * np.sin(bounce_angle)
            # Ball is moving left
            else:
                self.speed_x = speed * np.cos(bounce_angle)
                self.speed_y = -speed * np.sin(bounce_angle)
            
            # Slightly increase speed
            self.speed_x *= 1.1
            self.speed_y *= 1.1
            return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)

class PongGame:
    def __init__(self, ai_opponent=True):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pong AI")
        self.clock = pygame.time.Clock()
        
        # Create game objects
        self.player = Paddle(50)
        self.opponent = Paddle(WIDTH - 50 - PADDLE_WIDTH, is_ai=ai_opponent)
        self.ball = Ball()
        
        # Font for score
        self.font = pygame.font.Font(None, 74)
        
        self.game_over = False
        self.frame_count = 0

    def get_state(self):
        return np.array([
            self.ball.y / HEIGHT,  # Ball Y position
            self.ball.x / WIDTH,   # Ball X position
            self.ball.speed_y / BALL_SPEED,  # Ball Y velocity
            self.ball.speed_x / BALL_SPEED,  # Ball X velocity
            self.opponent.y / HEIGHT,  # AI paddle position
            self.player.y / HEIGHT,    # Player paddle position
        ])

    def reset(self):
        self.ball.reset()
        self.player.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.opponent.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.frame_count = 0
        return self.get_state()

    def step(self, action):
        reward = 0
        self.frame_count += 1
        
        # Move AI paddle based on action
        if action == 1:  # Move up
            self.opponent.move(up=True)
        elif action == 2:  # Move down
            self.opponent.move(up=False)
        
        # Store previous ball position
        prev_ball_x = self.ball.x
        
        # Move ball
        self.ball.move()
        
        # Check collisions
        hit_player = self.ball.paddle_collision(self.player)
        hit_opponent = self.ball.paddle_collision(self.opponent)
        
        # Reward for hitting the ball
        if hit_opponent:
            reward += 0.5
        
        # Reward for ball getting closer to player (when moving left)
        if self.ball.speed_x < 0 and self.ball.x < prev_ball_x:
            reward += 0.1
        
        # Score points
        done = False
        if self.ball.x <= 0:  # AI loses
            self.opponent.score += 1
            reward = -1
            done = True
        elif self.ball.x >= WIDTH:  # AI wins
            self.player.score += 1
            reward = 1
            done = True
            
        # Small penalty for not moving
        if action == 0:
            reward -= 0.01
            
        # Game over conditions
        if done:
            self.ball.reset()
        
        return self.get_state(), reward, done

    def render(self):
        self.screen.fill(BLACK)
        
        # Draw center line
        pygame.draw.aaline(self.screen, WHITE, (WIDTH//2, 0), (WIDTH//2, HEIGHT))
        
        # Draw paddles and ball
        self.player.draw(self.screen)
        self.opponent.draw(self.screen)
        self.ball.draw(self.screen)
        
        # Draw scores
        player_score = self.font.render(str(self.player.score), True, WHITE)
        opponent_score = self.font.render(str(self.opponent.score), True, WHITE)
        self.screen.blit(player_score, (WIDTH//4, 20))
        self.screen.blit(opponent_score, (3*WIDTH//4, 20))
        
        pygame.display.flip()

    def play_step(self, ai_agent=None):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, 0
            
        # Get player input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.player.move(up=True)
        if keys[pygame.K_DOWN]:
            self.player.move(up=False)
        
        # Get AI action if agent is provided
        if ai_agent is not None:
            state = self.get_state()
            action = ai_agent.act(state, training=False)
            _, reward, done = self.step(action)
        else:
            # Simple AI: Follow the ball
            if self.opponent.get_center() < self.ball.y:
                self.opponent.move(up=False)
            elif self.opponent.get_center() > self.ball.y:
                self.opponent.move(up=True)
            
            # Move ball and check for scoring
            self.ball.move()
            self.ball.paddle_collision(self.player)
            self.ball.paddle_collision(self.opponent)
            
            # Check scoring
            done = False
            if self.ball.x <= 0:
                self.opponent.score += 1
                self.ball.reset()
                done = True
            elif self.ball.x >= WIDTH:
                self.player.score += 1
                self.ball.reset()
                done = True
        
        self.render()
        self.clock.tick(60)
        
        return True, done

def main():
    game = PongGame(ai_opponent=True)
    running = True
    
    while running:
        running, _ = game.play_step()
    
    pygame.quit()

if __name__ == "__main__":
    main()
