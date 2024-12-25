import pygame
import random
import numpy as np
from enum import Enum
import os

# Initialize Pygame
pygame.init()

# Constants
WIDTH = 800
HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Colors
BACKGROUND = (15, 15, 35)
GRID_COLOR = (30, 30, 50)
SNAKE_COLOR = (50, 255, 150)
SNAKE_HEAD_COLOR = (30, 200, 100)
FOOD_COLOR = (255, 50, 50)
BORDER_COLOR = (40, 40, 60)
TEXT_COLOR = (200, 200, 220)

# Load or create high score
def load_high_score():
    if os.path.exists('high_score.txt'):
        with open('high_score.txt', 'r') as f:
            return int(f.read())
    return 0

def save_high_score(score):
    with open('high_score.txt', 'w') as f:
        f.write(str(score))

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class SnakeGame:
    def __init__(self, enable_render=True, speed=1):
        self.enable_render = enable_render
        self.speed = speed
        if enable_render:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Neural Snake")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        self.high_score = load_high_score()
        self.reset()

    def reset(self):
        # Initialize snake in the middle
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = Direction.RIGHT
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.max_steps = GRID_WIDTH * GRID_HEIGHT * 2
        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(1, GRID_WIDTH-2), random.randint(1, GRID_HEIGHT-2))
            if food not in self.snake:
                return food

    def _get_state(self):
        head = self.snake[0]
        
        # Danger straight, right, left relative to current direction
        danger_straight = self._is_collision(self._get_point_in_direction(head, self.direction))
        danger_right = self._is_collision(self._get_point_in_direction(head, self._get_right_direction()))
        danger_left = self._is_collision(self._get_point_in_direction(head, self._get_left_direction()))
        
        # Current direction
        dir_up = self.direction == Direction.UP
        dir_down = self.direction == Direction.DOWN
        dir_left = self.direction == Direction.LEFT
        dir_right = self.direction == Direction.RIGHT
        
        # Food direction
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        
        return np.array([
            danger_straight,
            danger_right,
            danger_left,
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            food_up,
            food_down,
            food_left,
            food_right
        ], dtype=int)

    def _is_collision(self, point):
        x, y = point
        # Check walls
        if x < 1 or x >= GRID_WIDTH-1 or y < 1 or y >= GRID_HEIGHT-1:
            return True
        # Check snake body
        if point in self.snake[1:]:
            return True
        return False

    def _get_point_in_direction(self, point, direction):
        x, y = point
        if direction == Direction.UP:
            return (x, y - 1)
        elif direction == Direction.DOWN:
            return (x, y + 1)
        elif direction == Direction.LEFT:
            return (x - 1, y)
        else:  # RIGHT
            return (x + 1, y)

    def _get_right_direction(self):
        if self.direction == Direction.UP:
            return Direction.RIGHT
        elif self.direction == Direction.RIGHT:
            return Direction.DOWN
        elif self.direction == Direction.DOWN:
            return Direction.LEFT
        else:  # LEFT
            return Direction.UP

    def _get_left_direction(self):
        if self.direction == Direction.UP:
            return Direction.LEFT
        elif self.direction == Direction.LEFT:
            return Direction.DOWN
        elif self.direction == Direction.DOWN:
            return Direction.RIGHT
        else:  # RIGHT
            return Direction.UP

    def _update_direction(self, action):
        # [straight, right, left]
        if action == 1:  # right
            self.direction = self._get_right_direction()
        elif action == 2:  # left
            self.direction = self._get_left_direction()
        # action == 0: keep going straight

    def step(self, action):
        self.steps += 1
        self._update_direction(action)
        
        # Move snake
        head = self.snake[0]
        new_head = self._get_point_in_direction(head, self.direction)
        
        # Check collision
        if self._is_collision(new_head):
            return self._get_state(), -10, True
        
        self.snake.insert(0, new_head)
        
        # Check food
        reward = 0
        if new_head == self.food:
            self.score += 1
            if self.score > self.high_score:
                self.high_score = self.score
                save_high_score(self.high_score)
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
            # Small reward for moving towards food
            if self._manhattan_distance(new_head, self.food) < self._manhattan_distance(head, self.food):
                reward = 0.1
            else:
                reward = -0.1
        
        # Check if game is won
        if len(self.snake) == (GRID_WIDTH-2) * (GRID_HEIGHT-2):
            return self._get_state(), 50, True
            
        # Check if max steps reached
        if self.steps >= self.max_steps:
            return self._get_state(), -5, True
            
        return self._get_state(), reward, False

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def render(self):
        if not self.enable_render:
            return
            
        self.screen.fill(BACKGROUND)
        
        # Draw grid
        for x in range(0, WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WIDTH, y))
            
        # Draw border
        pygame.draw.rect(self.screen, BORDER_COLOR, (0, 0, WIDTH, GRID_SIZE))  # Top
        pygame.draw.rect(self.screen, BORDER_COLOR, (0, HEIGHT-GRID_SIZE, WIDTH, GRID_SIZE))  # Bottom
        pygame.draw.rect(self.screen, BORDER_COLOR, (0, 0, GRID_SIZE, HEIGHT))  # Left
        pygame.draw.rect(self.screen, BORDER_COLOR, (WIDTH-GRID_SIZE, 0, GRID_SIZE, HEIGHT))  # Right
        
        # Draw snake with gradient effect
        for i, (x, y) in enumerate(self.snake):
            color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_COLOR
            alpha = 255 - min(255, (i * 5))  # Gradient effect
            s = pygame.Surface((GRID_SIZE-2, GRID_SIZE-2))
            s.fill(color)
            s.set_alpha(alpha)
            self.screen.blit(s, (x*GRID_SIZE+1, y*GRID_SIZE+1))
            
        # Draw food with glow effect
        food_x, food_y = self.food
        glow_radius = GRID_SIZE
        for radius in range(glow_radius, 0, -2):
            alpha = int(100 * (radius / glow_radius))
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*FOOD_COLOR, alpha), (radius, radius), radius)
            self.screen.blit(s, (food_x*GRID_SIZE + GRID_SIZE//2 - radius, 
                                food_y*GRID_SIZE + GRID_SIZE//2 - radius))
        pygame.draw.circle(self.screen, FOOD_COLOR, 
                         (food_x*GRID_SIZE + GRID_SIZE//2, 
                          food_y*GRID_SIZE + GRID_SIZE//2), GRID_SIZE//3)
        
        # Draw score and speed
        score_text = self.font.render(f'Score: {self.score}', True, TEXT_COLOR)
        high_score_text = self.font.render(f'High Score: {self.high_score}', True, TEXT_COLOR)
        speed_text = self.font.render(f'Speed: {self.speed}x', True, TEXT_COLOR)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(high_score_text, (WIDTH - high_score_text.get_width() - 10, 10))
        self.screen.blit(speed_text, (WIDTH//2 - speed_text.get_width()//2, 10))
        
        pygame.display.flip()
        self.clock.tick(10 * self.speed)  # Control game speed

def play_game():
    game = SnakeGame()
    running = True
    while running:
        action = 0  # Default to going straight
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            action = 1  # Turn right
        elif keys[pygame.K_LEFT]:
            action = 2  # Turn left
            
        _, reward, done = game.step(action)
        game.render()
        
        if done:
            print(f"Game Over! Score: {game.score}")
            game.reset()
            
    pygame.quit()

if __name__ == "__main__":
    play_game()
