import sys
import pygame
import rendering
import logic
import random
import time
import globalVariables

pygame.init()

clock = pygame.time.Clock()

globalVariables.screen = pygame.display.set_mode(globalVariables.size)
logic.updateApplePos()
pygame.display.set_caption('Snake')


update_interval = max(60-globalVariables.snake_speed, 0) # How many frames per update (assuming a fps of 60)
globalVariables.snake_list = logic.createSnakeList()


def step():
    logic.update()
    reward = 10

    rendering.render()
    window_pixel_matrix = pygame.surfarray.pixels3d(globalVariables.screen)
    
    return reward, window_pixel_matrix



# Game Loop
while 1:

    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                if globalVariables.snake_direction != 1:
                    globalVariables.pending_snake_direction = 3
            elif event.key == pygame.K_RIGHT:
                if globalVariables.snake_direction != 3:
                    globalVariables.pending_snake_direction = 1
            elif event.key == pygame.K_UP:
                if globalVariables.snake_direction != 2:
                    globalVariables.pending_snake_direction = 0
            elif event.key == pygame.K_DOWN:
                if globalVariables.snake_direction != 0:
                    globalVariables.pending_snake_direction = 2
    

    reward, screen = step()

    #clock.tick(globalVariables.fps) # Use for rendering and showing the game
    clock.tick() # Don't delay framerate when not rendering


pygame.quit()
quit()