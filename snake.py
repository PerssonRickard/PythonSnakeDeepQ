import logic
import rendering
import globalVariables
import torch
import numpy as np


def initalizeGame(pygame):
    # Set window properties
    globalVariables.screen = pygame.display.set_mode(globalVariables.size)
    pygame.display.set_caption('Snake')

    # Initialize logic
    logic.resetGame()

    # Render the initial frame
    rendering.render()

def step(action):
    
    # Take the selected action
    globalVariables.pending_snake_direction = action

    # Take step and receive reward
    reward, isTerminalState = logic.update()

    # Add reward to score
    globalVariables.score = globalVariables.score + reward

    # Render the next frame
    screen = rendering.render()

    # Update the number of steps counter
    globalVariables.numberOfSteps = globalVariables.numberOfSteps + 1

    return reward, screen, isTerminalState

def getGameFrame(pygame):
    window_pixel_matrix = pygame.surfarray.pixels3d(globalVariables.screen)
    screen = torch.from_numpy(np.copy(window_pixel_matrix))

    return screen