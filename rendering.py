import globalVariables
import pygame
import torch
import numpy as np
from PIL import Image, ImageTk
import torchvision.transforms as transforms

black = 0, 0, 0
white = 255, 255, 255
red = 255, 0, 0

def render(visualize):

    resetImage()
    renderApple()
    renderSnake()

    if visualize:
        # Update the label widget with the new frame
        imgTk = ImageTk.PhotoImage(globalVariables.image)
        globalVariables.screen.configure(image=imgTk)
        globalVariables.screen.image = imgTk

    tensorTransform = transforms.ToTensor()
    window_pixel_tensor = tensorTransform(globalVariables.image).permute(0, 2, 1).unsqueeze(1)

    #globalVariables.screen.fill(black)
    #renderApple()
    #renderSnake()
    #pygame.display.flip()

    #window_pixel_matrix = pygame.surfarray.pixels3d(globalVariables.screen)
    #window_pixel_matrix_tensor = torch.from_numpy(np.copy(window_pixel_matrix))

    return window_pixel_tensor

def resetImage():
    globalVariables.screenDraw.rectangle([0, 0, globalVariables.width, globalVariables.height], fill='black')

def renderSnake():
    for snake_body_pos in globalVariables.snake_list:
        x = snake_body_pos[0]*globalVariables.snake_block_size
        y = snake_body_pos[1]*globalVariables.snake_block_size
        globalVariables.screenDraw.rectangle([x, y, x + globalVariables.snake_block_size, y + globalVariables.snake_block_size], fill='white')

    '''
    for snake_body_pos in globalVariables.snake_list:
        pygame.draw.rect(globalVariables.screen, white, [snake_body_pos[0]*globalVariables.snake_block_size, 
            snake_body_pos[1]*globalVariables.snake_block_size, globalVariables.snake_block_size, globalVariables.snake_block_size])
    '''

def renderApple():
    x = globalVariables.apple_pos_x*globalVariables.snake_block_size
    y = globalVariables.apple_pos_y*globalVariables.snake_block_size
    globalVariables.screenDraw.rectangle([x, y, x + globalVariables.snake_block_size, y + globalVariables.snake_block_size], fill='red')

    #pygame.draw.rect(globalVariables.screen, red, [globalVariables.apple_pos_x*globalVariables.snake_block_size, 
    #        globalVariables.apple_pos_y*globalVariables.snake_block_size, globalVariables.snake_block_size, globalVariables.snake_block_size])