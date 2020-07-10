import globalVariables
import pygame
import torch
import numpy as np
from PIL import Image

black = 0, 0, 0
white = 255, 255, 255
red = 255, 0, 0

def render():
    globalVariables.screen.fill(black)
    renderApple()
    renderSnake()
    pygame.display.flip()

    window_pixel_matrix = pygame.surfarray.pixels3d(globalVariables.screen)
    window_pixel_matrix_tensor = torch.from_numpy(np.copy(window_pixel_matrix))

    return window_pixel_matrix_tensor

def renderSnake():
    for snake_body_pos in globalVariables.snake_list:
        pygame.draw.rect(globalVariables.screen, white, [snake_body_pos[0]*globalVariables.snake_block_size, 
            snake_body_pos[1]*globalVariables.snake_block_size, globalVariables.snake_block_size, globalVariables.snake_block_size])

def renderApple():
     pygame.draw.rect(globalVariables.screen, red, [globalVariables.apple_pos_x*globalVariables.snake_block_size, 
            globalVariables.apple_pos_y*globalVariables.snake_block_size, globalVariables.snake_block_size, globalVariables.snake_block_size])