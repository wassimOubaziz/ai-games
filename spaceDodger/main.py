import pygame
import sys
import random
import math

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 600
SHIP_SIZE = 40
ASTEROID_SIZE = 30
FPS = 60
AI_SPEED_MULTIPLIER = 1

# Set up some colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 191, 255)
YELLOW = (255, 255, 0)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Dodger")

# Set up the fonts
font = pygame.font.Font(None, 36)

# Set up the clock
clock = pygame.time.Clock()

class Spaceship:
    def __init__(self):
        self.x = 100
        self.y = HEIGHT // 2
        self.velocity = 0
        self.thrust = 0.5
        self.max_velocity = 8
        
        # Create a triangle for the spaceship
        self.points = [
            (self.x, self.y - SHIP_SIZE//2),  # Top
            (self.x - SHIP_SIZE//2, self.y + SHIP_SIZE//2),  # Bottom left
            (self.x + SHIP_SIZE//2, self.y),  # Right
            (self.x - SHIP_SIZE//2, self.y - SHIP_SIZE//2)   # Top left
        ]

    def update(self):
        self.y += self.velocity
        
        # Apply thrust (gravity)
        if abs(self.velocity) < self.max_velocity:
            self.velocity += self.thrust
            
        # Update ship points
        self.points = [
            (self.x, self.y - SHIP_SIZE//2),
            (self.x - SHIP_SIZE//2, self.y + SHIP_SIZE//2),
            (self.x + SHIP_SIZE//2, self.y),
            (self.x - SHIP_SIZE//2, self.y - SHIP_SIZE//2)
        ]

    def thrust_up(self):
        self.velocity = -6

    def draw(self):
        # Draw the ship body
        pygame.draw.polygon(screen, WHITE, self.points)
        
        # Draw thrust effect when moving up
        if self.velocity < 0:
            thrust_points = [
                (self.x - SHIP_SIZE//2, self.y + SHIP_SIZE//2),
                (self.x - SHIP_SIZE//2 - 10, self.y + SHIP_SIZE//2 + 10),
                (self.x - SHIP_SIZE//4, self.y + SHIP_SIZE//2)
            ]
            pygame.draw.polygon(screen, YELLOW, thrust_points)

    def get_hitbox(self):
        return pygame.Rect(self.x - SHIP_SIZE//2, self.y - SHIP_SIZE//2, SHIP_SIZE, SHIP_SIZE)

class Asteroid:
    def __init__(self, x=None, difficulty_multiplier=1.0):
        self.x = x if x is not None else WIDTH + ASTEROID_SIZE
        self.y = random.randint(ASTEROID_SIZE, HEIGHT - ASTEROID_SIZE)
        base_speed = random.uniform(4, 7)
        self.speed = base_speed * difficulty_multiplier
        self.rotation = 0
        self.rotation_speed = random.uniform(-3, 3)
        self.passed = False
        self.size_multiplier = min(1.0 + (difficulty_multiplier - 1) * 0.3, 1.5)  # Size increases with difficulty, max 150%
        self.points = self._generate_points()

    def _generate_points(self):
        num_points = 8
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            radius = ASTEROID_SIZE * self.size_multiplier * random.uniform(0.8, 1.2)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append((x, y))
        return points

    def update(self, speed_multiplier=1):
        self.x -= self.speed * speed_multiplier
        self.rotation += self.rotation_speed

    def draw(self):
        # Rotate and translate points
        rotated_points = []
        for x, y in self.points:
            # Rotate
            rot_x = x * math.cos(self.rotation) - y * math.sin(self.rotation)
            rot_y = x * math.sin(self.rotation) + y * math.cos(self.rotation)
            # Translate
            rotated_points.append((rot_x + self.x, rot_y + self.y))
        
        pygame.draw.polygon(screen, WHITE, rotated_points)

    def get_hitbox(self):
        hitbox_size = ASTEROID_SIZE * 2 * self.size_multiplier
        return pygame.Rect(self.x - hitbox_size/2, self.y - hitbox_size/2, 
                         hitbox_size, hitbox_size)

def draw_stars(num_stars=100):
    for _ in range(num_stars):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        pygame.draw.circle(screen, WHITE, (x, y), 1)

def main(ai_mode=False):
    from learning import SpaceDodgerAI
    
    ship = Spaceship()
    asteroids = [Asteroid()]
    score = 0
    game_over = False
    frames_alive = 0
    frame_count = 0
    stars_positions = [(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(100)]
    
    # Difficulty progression
    difficulty_multiplier = 1.0
    difficulty_increase_rate = 0.1  # Increase every 10 points
    asteroid_spawn_distance = 300  # Initial spawn distance between asteroids
    min_spawn_distance = 200       # Minimum spawn distance at max difficulty
    
    # Initialize AI if in AI mode
    ai_agent = SpaceDodgerAI("best_model.pth") if ai_mode else None
    episode = 0
    best_score = 0
    speed_multiplier = AI_SPEED_MULTIPLIER if ai_mode else 1
    
    # Speed control
    paused = True
    show_stats = True
    training_mode = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if ai_mode and score > best_score:
                    ai_agent.save("best_model.pth")
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not ai_mode:
                    if game_over:
                        ship = Spaceship()
                        asteroids = [Asteroid()]
                        score = 0
                        difficulty_multiplier = 1.0
                        asteroid_spawn_distance = 300
                        game_over = False
                        frames_alive = 0
                    else:
                        ship.thrust_up()
                elif event.key == pygame.K_p and ai_mode:
                    paused = not paused
                elif event.key == pygame.K_s and ai_mode:
                    show_stats = not show_stats
                elif event.key == pygame.K_t and ai_mode:
                    training_mode = not training_mode
                    print(f"Training mode: {'ON' if training_mode else 'OFF'}")
                elif event.key == pygame.K_UP and ai_mode:
                    speed_multiplier = min(speed_multiplier + 1, 10)
                elif event.key == pygame.K_DOWN and ai_mode:
                    speed_multiplier = max(speed_multiplier - 1, 1)

        if not game_over and not paused:
            frame_count += 1
            
            # Update difficulty based on score
            difficulty_multiplier = 1.0 + (score * difficulty_increase_rate)
            asteroid_spawn_distance = max(min_spawn_distance, 
                                       300 - (score * 2))  # Decrease spawn distance as score increases
            
            # AI control
            if ai_mode and frame_count % 4 == 0:
                state = ai_agent.get_state(ship, asteroids)
                action = ai_agent.act(state, training=training_mode)
                if action == 1:
                    ship.thrust_up()

            # Handle manual input
            if not ai_mode:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    ship.thrust_up()

            ship.update()
            frames_alive += 1

            # Calculate reward for AI
            reward = 0.1  # Small reward for staying alive

            # Check boundaries
            if ship.y < 0 or ship.y > HEIGHT:
                game_over = True
                reward = -10 if ai_mode else 0

            # Update and check asteroids
            for asteroid in asteroids:
                asteroid.update(speed_multiplier)
                
                # Check collision
                if asteroid.get_hitbox().colliderect(ship.get_hitbox()):
                    game_over = True
                    reward = -10 if ai_mode else 0
                
                # Score point if ship passes asteroid
                if not asteroid.passed and ship.x > asteroid.x + ASTEROID_SIZE:
                    asteroid.passed = True
                    score += 1
                    reward = 1 if ai_mode else 0

            # Remove off-screen asteroids
            asteroids = [ast for ast in asteroids if ast.x > -ASTEROID_SIZE * 2]

            # Add new asteroid
            if len(asteroids) == 0 or asteroids[-1].x < WIDTH - asteroid_spawn_distance:
                asteroids.append(Asteroid(difficulty_multiplier=difficulty_multiplier))

            # AI Learning
            if ai_mode and training_mode and frame_count % 4 == 0:
                next_state = ai_agent.get_state(ship, asteroids)
                ai_agent.remember(state, action, reward, next_state, game_over)
                ai_agent.replay()

        # Drawing
        screen.fill(BLACK)
        
        # Draw stars (static background)
        for star_x, star_y in stars_positions:
            pygame.draw.circle(screen, WHITE, (int(star_x), int(star_y)), 1)
        
        ship.draw()
        for asteroid in asteroids:
            asteroid.draw()

        # Draw score and difficulty
        score_text = font.render(f"Score: {score}", True, WHITE)
        difficulty_text = font.render(f"Difficulty: {difficulty_multiplier:.1f}x", True, 
                                    (min(255, int(difficulty_multiplier * 100)), 
                                     max(0, int(255 - difficulty_multiplier * 50)), 0))
        screen.blit(score_text, (10, 10))
        screen.blit(difficulty_text, (WIDTH - 200, 10))

        if ai_mode and show_stats:
            # Draw AI stats
            episode_text = font.render(f"Episode: {episode}", True, WHITE)
            best_text = font.render(f"Best Score: {best_score}", True, WHITE)
            speed_text = font.render(f"Speed: {speed_multiplier}x", True, GREEN)
            mode_text = font.render(f"Training: {'ON' if training_mode else 'OFF'}", True, GREEN if training_mode else RED)
            screen.blit(episode_text, (10, 40))
            screen.blit(best_text, (10, 70))
            screen.blit(speed_text, (10, 100))
            screen.blit(mode_text, (10, 130))
            
            if paused:
                pause_text = font.render("PAUSED", True, RED)
                text_rect = pause_text.get_rect(center=(WIDTH/2, HEIGHT/2))
                screen.blit(pause_text, text_rect)

        # Handle game over
        if game_over:
            if ai_mode:
                # Save the model if it achieved a new best score
                if score > best_score:
                    best_score = score
                    ai_agent.save("best_model.pth")
                
                # Reset for next episode
                episode += 1
                ship = Spaceship()
                asteroids = [Asteroid()]
                score = 0
                difficulty_multiplier = 1.0
                asteroid_spawn_distance = 300
                game_over = False
                frames_alive = 0
            else:
                game_over_text = font.render("Game Over! Press SPACE to restart", True, WHITE)
                text_rect = game_over_text.get_rect(center=(WIDTH/2, HEIGHT/2))
                screen.blit(game_over_text, text_rect)

        pygame.display.flip()
        clock.tick(FPS * speed_multiplier)

if __name__ == "__main__":
    import sys
    ai_mode = "--ai" in sys.argv
    main(ai_mode)
