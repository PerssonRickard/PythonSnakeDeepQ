import globalVariables
import pygame

black = 0, 0, 0
white = 255, 255, 255
red = 255, 0, 0

def render():
    globalVariables.screen.fill(black)
    renderApple()
    renderSnake()
    pygame.display.flip()

def renderSnake():
    for snake_body_pos in globalVariables.snake_list:
        pygame.draw.rect(globalVariables.screen, white, [snake_body_pos[0]*globalVariables.snake_block_size, 
            snake_body_pos[1]*globalVariables.snake_block_size, globalVariables.snake_block_size, globalVariables.snake_block_size])

def renderApple():
     pygame.draw.rect(globalVariables.screen, red, [globalVariables.apple_pos_x*globalVariables.snake_block_size, 
            globalVariables.apple_pos_y*globalVariables.snake_block_size, globalVariables.snake_block_size, globalVariables.snake_block_size])