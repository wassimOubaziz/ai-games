import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 640, 480
PIPE_WIDTH = 80
PIPE_GAP = 200  # Gap between top and bottom pipes
BIRD_SIZE = 30
FPS = 60
AI_SPEED_MULTIPLIER = 4  # Speed up factor for AI training

# Set up some colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the fonts
font = pygame.font.Font(None, 36)

# Set up the clock
clock = pygame.time.Clock()

class Bird:
    def __init__(self):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.velocity = 0

    def update(self):
        self.y += self.velocity
        self.velocity += 0.5

    def draw(self):
        pygame.draw.rect(screen, RED, (self.x, self.y, BIRD_SIZE, BIRD_SIZE))

class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(100, HEIGHT - 100 - PIPE_GAP)
        self.passed = False

    def update(self, speed_multiplier=1):
        self.x -= 5 * speed_multiplier  # Increased base speed and apply multiplier

    def draw(self):
        # Draw top pipe
        pygame.draw.rect(screen, WHITE, (self.x, 0, PIPE_WIDTH, self.gap_y))
        # Draw bottom pipe
        pygame.draw.rect(screen, WHITE, (self.x, self.gap_y + PIPE_GAP, PIPE_WIDTH, HEIGHT - (self.gap_y + PIPE_GAP)))

    def collides_with_bird(self, bird):
        bird_rect = pygame.Rect(bird.x, bird.y, BIRD_SIZE, BIRD_SIZE)
        top_pipe = pygame.Rect(self.x, 0, PIPE_WIDTH, self.gap_y)
        bottom_pipe = pygame.Rect(self.x, self.gap_y + PIPE_GAP, PIPE_WIDTH, HEIGHT - (self.gap_y + PIPE_GAP))
        
        return bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe)

def main(ai_mode=False):
    from learning import FlappyBirdAI
    
    bird = Bird()
    pipes = [Pipe(WIDTH)]
    score = 0
    game_over = False
    frames_alive = 0
    speed_multiplier = AI_SPEED_MULTIPLIER if ai_mode else 1
    frame_count = 0
    
    # Initialize AI if in AI mode
    ai_agent = FlappyBirdAI("best_model.pth") if ai_mode else None
    episode = 0
    best_score = 0
    
    # Speed control
    paused = False
    show_stats = True
    training_mode = True  # New flag to control whether AI is training or just playing

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
                        bird = Bird()
                        pipes = [Pipe(WIDTH)]
                        score = 0
                        game_over = False
                        frames_alive = 0
                    else:
                        bird.velocity = -8
                elif event.key == pygame.K_p and ai_mode:  # Pause/Unpause with P key
                    paused = not paused
                elif event.key == pygame.K_s and ai_mode:  # Toggle stats with S key
                    show_stats = not show_stats
                elif event.key == pygame.K_t and ai_mode:  # Toggle training mode with T key
                    training_mode = not training_mode
                    print(f"Training mode: {'ON' if training_mode else 'OFF'}")
                elif event.key == pygame.K_UP and ai_mode:  # Increase speed
                    speed_multiplier = min(speed_multiplier + 1, 10)
                elif event.key == pygame.K_DOWN and ai_mode:  # Decrease speed
                    speed_multiplier = max(speed_multiplier - 1, 1)

        if not game_over and not paused:
            frame_count += 1
            
            # AI control
            if ai_mode and frame_count % 4 == 0:  # Frame skipping for more consistent behavior
                state = ai_agent.get_state(bird, pipes)
                action = ai_agent.act(state, training=training_mode)
                if action == 1:  # Jump
                    bird.velocity = -8

            bird.update()
            frames_alive += 1

            # Calculate reward for AI
            reward = 0.1  # Small reward for staying alive

            # Check boundaries
            if bird.y < 0 or bird.y > HEIGHT - BIRD_SIZE:
                game_over = True
                reward = -10 if ai_mode else 0

            # Update and check pipes
            for pipe in pipes:
                pipe.update(speed_multiplier)
                
                # Check collision
                if pipe.collides_with_bird(bird):
                    game_over = True
                    reward = -10 if ai_mode else 0
                
                # Score point if bird passes pipe
                if not pipe.passed and bird.x > pipe.x + PIPE_WIDTH:
                    pipe.passed = True
                    score += 1
                    reward = 1 if ai_mode else 0

            # Remove off-screen pipes
            pipes = [pipe for pipe in pipes if pipe.x > -PIPE_WIDTH]

            # Add new pipe
            if len(pipes) == 0 or pipes[-1].x < WIDTH - 300:
                pipes.append(Pipe(WIDTH))

            # AI Learning
            if ai_mode and training_mode and frame_count % 4 == 0:
                next_state = ai_agent.get_state(bird, pipes)
                ai_agent.remember(state, action, reward, next_state, game_over)
                ai_agent.replay()

        # Drawing
        screen.fill(BLACK)
        bird.draw()
        for pipe in pipes:
            pipe.draw()

        # Draw score
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

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
                bird = Bird()
                pipes = [Pipe(WIDTH)]
                score = 0
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