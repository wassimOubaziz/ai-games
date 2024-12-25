import pygame
import random
import numpy as np
import math
from enum import Enum
import os

# Initialize Pygame
pygame.init()

# Constants
WIDTH = 800
HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLORS = {
    'red': (255, 50, 50),
    'blue': (50, 50, 255),
    'green': (50, 255, 50),
    'yellow': (255, 255, 50)
}

# Player properties
PLAYER_SIZE = 30
GRAVITY = 0.8
JUMP_FORCE = -15
MOVE_SPEED = 7
MAX_JUMPS = 2

class ColorState(Enum):
    RED = 'red'
    BLUE = 'blue'
    GREEN = 'green'
    YELLOW = 'yellow'

class Platform:
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.original_y = y
        self.move_range = 0
        self.move_speed = 0
        self.time = random.random() * math.pi * 2
        self.safe_zone = True  # All platforms are safe zones by default

    def update(self):
        if self.move_range > 0:
            self.time += 0.03  # Slower movement
            self.rect.y = self.original_y + math.sin(self.time) * self.move_range

    def draw(self, screen, camera_y):
        adjusted_rect = self.rect.copy()
        adjusted_rect.y -= camera_y
        
        # Create glow effect
        for i in range(3):
            glow_rect = adjusted_rect.copy()
            glow_rect.inflate_ip(i*2, i*2)
            color = list(COLORS[self.color])
            for j in range(3):
                color[j] = min(255, color[j] + 20)
            pygame.draw.rect(screen, color, glow_rect, border_radius=5)
        
        pygame.draw.rect(screen, COLORS[self.color], adjusted_rect, border_radius=5)

class Particle:
    def __init__(self, x, y, color, velocity_x, velocity_y):
        self.x = x
        self.y = y
        self.color = color
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.alpha = 255
        self.size = random.randint(3, 6)
        
    def update(self):
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.velocity_y += GRAVITY * 0.1
        self.alpha -= 5
        return self.alpha > 0
        
    def draw(self, screen, camera_y):
        if self.alpha > 0:
            color = list(COLORS[self.color])
            s = pygame.Surface((self.size, self.size))
            s.fill(color)
            s.set_alpha(self.alpha)
            screen.blit(s, (self.x, self.y - camera_y))

class ColorShiftGame:
    def __init__(self, enable_render=True):
        self.enable_render = enable_render
        if enable_render:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("ColorShift")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)  # Add small font for color indicators
        
        self.reset()
        
    def reset(self):
        self.player_pos = pygame.Vector2(WIDTH // 2, HEIGHT - 100)
        self.player_vel = pygame.Vector2(0, 0)
        self.current_color = ColorState.RED
        self.score = 0
        self.camera_y = 0
        self.particles = []
        self.platforms = self._generate_platforms()
        self.game_over = False
        self.jumps_left = MAX_JUMPS
        return self._get_state()
    
    def _generate_platforms(self):
        platforms = []
        # Ground platform
        platforms.append(Platform(0, HEIGHT - 40, WIDTH, 40, 'red'))
        
        # Generate platforms up to 10000 pixels high
        y = HEIGHT - 200
        while y > -10000:
            # Create wider platforms
            width = random.randint(150, 350)  # Increased minimum width
            x = random.randint(0, WIDTH - width)
            color = random.choice(list(ColorState)).value
            platform = Platform(x, y, width, 20, color)
            
            # Ensure platforms are reachable
            min_gap = 100  # Reduced minimum gap
            max_gap = 180  # Reduced maximum gap
            if platforms[-1].rect.right < x:  # If there's a gap on the left
                # Add a platform to bridge the gap
                bridge_x = platforms[-1].rect.right + random.randint(20, 60)
                bridge_width = x - bridge_x - random.randint(20, 40)
                if bridge_width > 50:  # Only add if there's enough space
                    bridge = Platform(bridge_x, y + random.randint(-20, 20), 
                                   bridge_width, 20, random.choice(list(ColorState)).value)
                    platforms.append(bridge)
            
            # Add movement to fewer platforms
            if random.random() < 0.2:  # Reduced from 0.3
                platform.move_range = random.randint(20, 80)  # Reduced range
                platform.move_speed = random.uniform(0.01, 0.03)  # Slower movement
            
            platforms.append(platform)
            y -= random.randint(60, 120)  # Reduced vertical gap further
        
        return platforms

    def _get_state(self):
        # Find closest platforms
        visible_platforms = [p for p in self.platforms if abs(p.rect.centery - (self.player_pos.y - self.camera_y)) < HEIGHT]
        visible_platforms.sort(key=lambda p: abs(p.rect.centery - (self.player_pos.y - self.camera_y)))
        
        state = []
        # Player info
        state.extend([
            self.player_pos.x / WIDTH,
            (self.player_pos.y - self.camera_y) / HEIGHT,
            self.player_vel.x / MOVE_SPEED,
            self.player_vel.y / JUMP_FORCE,
            [self.current_color == c for c in ColorState].index(True) / len(ColorState)
        ])
        
        # Nearest platforms (up to 3)
        for i in range(3):
            if i < len(visible_platforms):
                p = visible_platforms[i]
                state.extend([
                    p.rect.centerx / WIDTH,
                    (p.rect.centery - (self.player_pos.y - self.camera_y)) / HEIGHT,
                    p.rect.width / WIDTH,
                    [p.color == c.value for c in ColorState].index(True) / len(ColorState)
                ])
            else:
                state.extend([0, 0, 0, 0])
        
        return np.array(state)

    def step(self, action):
        # Action: [0: nothing, 1: left, 2: right, 3: jump, 4-7: color changes]
        reward = 0
        
        # Handle color changes
        if 4 <= action <= 7:
            new_color = list(ColorState)[action-4]
            if new_color != self.current_color:
                # Check if on a platform (safe zone)
                on_platform = False
                player_rect = pygame.Rect(self.player_pos.x - PLAYER_SIZE/2,
                                        self.player_pos.y - PLAYER_SIZE/2,
                                        PLAYER_SIZE, PLAYER_SIZE)
                
                for platform in self.platforms:
                    if (player_rect.bottom >= platform.rect.top and
                        player_rect.bottom <= platform.rect.top + 20 and
                        player_rect.right > platform.rect.left and
                        player_rect.left < platform.rect.right):
                        on_platform = True
                        break
                
                # Always allow color change, but only fall if not on any platform
                self.current_color = new_color
                self._add_color_change_particles()
                
                # Give small reward for successful color change
                reward += 0.1
        
        # Handle jump
        if action == 3:
            if self._is_grounded() or self.jumps_left > 0:
                self.player_vel.y = JUMP_FORCE
                self.jumps_left = max(0, self.jumps_left - 1)
                self._add_jump_particles()
                reward += 0.1  # Small reward for jumping
        
        # Apply physics
        self.player_vel.y += GRAVITY
        self.player_pos += self.player_vel
        
        # Update platforms and particles
        for platform in self.platforms:
            platform.update()
        
        self.particles = [p for p in self.particles if p.update()]
        
        # Handle collisions
        self._handle_collisions()
        
        # Update camera with smoother movement
        target_camera_y = self.player_pos.y - HEIGHT * 0.6
        self.camera_y += (target_camera_y - self.camera_y) * 0.1
        
        # Calculate reward
        height_score = -self.player_pos.y / 100
        if height_score > self.score:
            reward += height_score - self.score
            self.score = height_score
        
        # Check game over
        if self.player_pos.y - self.camera_y > HEIGHT:
            return self._get_state(), -10, True
            
        if self.player_pos.x < 0 or self.player_pos.x > WIDTH:
            self.player_pos.x = max(0, min(self.player_pos.x, WIDTH))  # Bounce from walls
            self.player_vel.x = 0
            reward -= 0.1  # Small penalty for hitting walls
        
        return self._get_state(), reward, False

    def _is_grounded(self):
        player_rect = pygame.Rect(self.player_pos.x - PLAYER_SIZE/2,
                                self.player_pos.y - PLAYER_SIZE/2,
                                PLAYER_SIZE, PLAYER_SIZE)
        
        for platform in self.platforms:
            if (platform.color == self.current_color.value and
                player_rect.bottom >= platform.rect.top and
                player_rect.bottom <= platform.rect.top + 20 and
                player_rect.right > platform.rect.left and
                player_rect.left < platform.rect.right):
                self.jumps_left = MAX_JUMPS  # Reset jumps when grounded
                return True
        return False

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - PLAYER_SIZE/2,
                                self.player_pos.y - PLAYER_SIZE/2,
                                PLAYER_SIZE, PLAYER_SIZE)
        
        for platform in self.platforms:
            # Check collision with any platform when in safe zone
            if platform.safe_zone and player_rect.colliderect(platform.rect):
                # Top collision
                if self.player_vel.y > 0 and player_rect.bottom >= platform.rect.top:
                    self.player_pos.y = platform.rect.top - PLAYER_SIZE/2
                    self.player_vel.y = 0
                    self.jumps_left = MAX_JUMPS
                # Bottom collision
                elif self.player_vel.y < 0 and player_rect.top <= platform.rect.bottom:
                    self.player_pos.y = platform.rect.bottom + PLAYER_SIZE/2
                    self.player_vel.y = 0
            # Check collision only with matching color when not in safe zone
            elif platform.color == self.current_color.value and player_rect.colliderect(platform.rect):
                # Top collision
                if self.player_vel.y > 0 and player_rect.bottom >= platform.rect.top:
                    self.player_pos.y = platform.rect.top - PLAYER_SIZE/2
                    self.player_vel.y = 0
                    self.jumps_left = MAX_JUMPS
                # Bottom collision
                elif self.player_vel.y < 0 and player_rect.top <= platform.rect.bottom:
                    self.player_pos.y = platform.rect.bottom + PLAYER_SIZE/2
                    self.player_vel.y = 0

    def _add_jump_particles(self):
        for _ in range(10):
            velocity_x = random.uniform(-2, 2)
            velocity_y = random.uniform(0, 2)
            self.particles.append(Particle(
                self.player_pos.x,
                self.player_pos.y + PLAYER_SIZE/2,
                self.current_color.value,
                velocity_x,
                velocity_y
            ))

    def _add_color_change_particles(self):
        for _ in range(20):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(2, 5)
            velocity_x = math.cos(angle) * speed
            velocity_y = math.sin(angle) * speed
            self.particles.append(Particle(
                self.player_pos.x,
                self.player_pos.y,
                self.current_color.value,
                velocity_x,
                velocity_y
            ))

    def render(self):
        if not self.enable_render:
            return
            
        self.screen.fill(BLACK)
        
        # Draw platforms
        for platform in self.platforms:
            platform.draw(self.screen, self.camera_y)
        
        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen, self.camera_y)
        
        # Draw player with glow effect
        player_center = (int(self.player_pos.x), int(self.player_pos.y - self.camera_y))
        for radius in range(PLAYER_SIZE//2 + 6, PLAYER_SIZE//2 - 1, -2):
            color = list(COLORS[self.current_color.value])
            for i in range(3):
                color[i] = min(255, color[i] + (PLAYER_SIZE//2 + 6 - radius) * 20)
            pygame.draw.circle(self.screen, color, player_center, radius)
        
        # Draw score
        score_text = self.font.render(f'Score: {int(self.score)}', True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Draw jumps left indicator
        jumps_text = self.font.render(f'Jumps: {self.jumps_left}', True, WHITE)
        self.screen.blit(jumps_text, (10, 50))
        
        # Draw color indicators
        color_y = 10
        for i, color_state in enumerate(ColorState):
            color = COLORS[color_state.value]
            indicator_text = self.small_font.render(f'{i+1}: {color_state.value.capitalize()}', True, color)
            self.screen.blit(indicator_text, (WIDTH - 120, color_y))
            # Highlight current color
            if color_state == self.current_color:
                pygame.draw.rect(self.screen, color, 
                               (WIDTH - 125, color_y, 120, 20), 1)
            color_y += 25
        
        pygame.display.flip()
        self.clock.tick(FPS)

def play_game():
    game = ColorShiftGame()
    running = True
    last_jump = 0  # Track last jump time
    
    while running:
        keys = pygame.key.get_pressed()
        action = 0  # Default to no action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Handle jump on key press (not hold)
                if event.key == pygame.K_SPACE:
                    current_time = pygame.time.get_ticks()
                    if current_time - last_jump > 200:  # 200ms cooldown between jumps
                        action = 3
                        last_jump = current_time
                # Handle color changes
                elif event.key == pygame.K_1:
                    action = 4
                elif event.key == pygame.K_2:
                    action = 5
                elif event.key == pygame.K_3:
                    action = 6
                elif event.key == pygame.K_4:
                    action = 7
        
        # Handle movement separately from other actions
        if keys[pygame.K_LEFT]:
            game.player_vel.x = -MOVE_SPEED
        elif keys[pygame.K_RIGHT]:
            game.player_vel.x = MOVE_SPEED
        else:
            game.player_vel.x = 0
        
        # Process the action (jump or color change)
        if action > 0:
            _, reward, done = game.step(action)
        else:
            # Still need to update physics and check collisions
            _, reward, done = game.step(0)
        
        game.render()
        
        if done:
            print(f"Game Over! Score: {int(game.score)}")
            game.reset()
            
    pygame.quit()

if __name__ == "__main__":
    play_game()
