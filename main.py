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

tick_count = 0

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



    if tick_count >= update_interval:
        logic.update()
        tick_count = 0

    rendering.render()

    tick_count += 1
    clock.tick(globalVariables.fps)


pygame.quit()
quit()