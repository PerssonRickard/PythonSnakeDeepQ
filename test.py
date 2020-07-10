import sys
import pygame
import rendering
import logic
import random
import time
import globalVariables
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.transform import resize

pygame.init()

white = 255, 255, 255
clock = pygame.time.Clock()

globalVariables.screen = pygame.display.set_mode(globalVariables.size)
pygame.display.set_caption('Snake')

while 1:

    #pygame.draw.rect(globalVariables.screen, white, [10, 10, 20, 20])

    img = Image.new('RGBA', (500, 500), color = 'red')
    mode = img.mode
    size = img.size
    data = img.tobytes()
    image = pygame.image.fromstring(data, size, mode)
    globalVariables.screen.blit(image, (0, 0))

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    clock.tick(globalVariables.fps)


pygame.quit()