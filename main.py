import sys
#import pygame
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
#from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from collections import deque
from qLearning import ReplayMemory
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk, ImageDraw


running = True
globalVariables.window = tk.Tk()
#pygame.init()

#clock = pygame.time.Clock()

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

#globalVariables.screen = pygame.display.set_mode(globalVariables.size)
logic.updateApplePos()
globalVariables.window.title('Snake')

def handle_closing_event():
    global running
    running = False

globalVariables.window.protocol("WM_DELETE_WINDOW", handle_closing_event)
#pygame.display.set_caption('Snake')

globalVariables.image = Image.new('L', globalVariables.size, color='black')
globalVariables.screenDraw = ImageDraw.Draw(globalVariables.image)
imageTk = ImageTk.PhotoImage(globalVariables.image)
globalVariables.screen = Label(globalVariables.window, image=imageTk)
globalVariables.screen.grid(row=0, column=0)
globalVariables.screen.pack()

globalVariables.snake_list = logic.createSnakeList()

visualize = False

# Render the initial frame
rendering.render(visualize)


def initializeState():

    #window_pixel_matrix = pygame.surfarray.pixels3d(globalVariables.screen)
    #screen = torch.from_numpy(np.copy(window_pixel_matrix))

    #resizedScreen = preprocessInput(screen)


    '''

    stackedFrames = torch.empty(1, 0, globalVariables.downSampleWidth, globalVariables.downSampleHeight)
    for i in range(4):
        stackedFrames = torch.cat((stackedFrames, resizedScreen.detach().clone()), 1)
    
    '''

    stackedFrames = torch.tensor(np.zeros((1, 3, globalVariables.width, globalVariables.height)), dtype=torch.float32)

    tensorTransform = transforms.ToTensor()
    window_pixel_tensor = tensorTransform(globalVariables.image).permute(0, 2, 1).unsqueeze(1)

    #test = resizedScreen.detach().clone()
    stackedFrames = torch.cat((window_pixel_tensor, stackedFrames), 1)

    return stackedFrames

def updateState(stackedFrames, frame):
    stackedFrames = torch.roll(stackedFrames, 1, dims=1)
    stackedFrames[:,0,:,:] = frame

    return stackedFrames


def preprocessInput(frame):

    # Convert to gray-scale
    frame = 0.2989*frame[:, :, 0] + 0.5870*frame[:, :, 1] + 0.1140*frame[:, :, 2]

    # Add dimensions and convert to float
    frame = frame.permute(1, 0).unsqueeze(0).unsqueeze(1).float()

    # Down-sample the image
    frame = F.interpolate(frame, size=(globalVariables.downSampleWidth, globalVariables.downSampleHeight), mode = 'bilinear')

    return frame

def plotFrame(frame):

    if frame.size(1) == 1:
        plt.imshow(frame.squeeze(), cmap='gray')
        plt.show()
    else:
        fig=plt.figure(figsize=(1, 4))

        fig.add_subplot(1, 4, 1)
        plt.imshow(frame[:,0,:,:].squeeze(), cmap='gray')
        fig.add_subplot(1, 4, 2)
        plt.imshow(frame[:,1,:,:].squeeze(), cmap='gray')
        fig.add_subplot(1, 4, 3)
        plt.imshow(frame[:,2,:,:].squeeze(), cmap='gray')
        fig.add_subplot(1, 4, 4)
        plt.imshow(frame[:,3,:,:].squeeze(), cmap='gray')

        plt.show()



stackedFrames = initializeState()


class deepQNetwork(nn.Module):
    def __init__(self):
        super(deepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(32*13*10, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32*13*10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the Q-networks
globalVariables.deepQNetwork1 = deepQNetwork()
globalVariables.deepQNetwork2 = deepQNetwork()
globalVariables.deepQNetwork1.to(device)
globalVariables.deepQNetwork2.to(device)

if globalVariables.pretrained:
    globalVariables.deepQNetwork1.load_state_dict(torch.load('C:/Users/Rickard/Documents/python/MachineLearning/PythonSnakeDeepQ/savedNetworks/network12times8OptimalRewards2.pt'))

# Set the starting parameters to be the same for both networks
globalVariables.deepQNetwork2.load_state_dict(globalVariables.deepQNetwork1.state_dict())

optimizer1 = optim.SGD(globalVariables.deepQNetwork1.parameters(), lr=globalVariables.learningRate, momentum=0.9)
optimizer2 = optim.SGD(globalVariables.deepQNetwork1.parameters(), lr=globalVariables.learningRate, momentum=0.9)

lossFunction = nn.MSELoss() # Use reduction = sum ?

def greedy(state):

    state = state.to(device)
    with torch.no_grad():
        if globalVariables.deepQNetwork1Frozen:
            qValues = globalVariables.deepQNetwork2(state)
        else:
            qValues = globalVariables.deepQNetwork1(state)
        action = np.argmax(qValues.cpu().detach().numpy())

    return action, qValues


def epsilonGreedy(state, epsilon):

    assert(epsilon <= 1)

    random_value = np.random.uniform()

    greedyAction, qValues = greedy(state)
    if random_value < epsilon:
        action = np.random.randint(0, 4)
    else:
        action = greedyAction

    return action, qValues



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

    globalVariables.epsilon = max(globalVariables.epsilon - stepEpsilon, endingEpsilon)

    return reward, screen, isTerminalState

def calculateQLearningTargets(sampledMiniBatch):

    with torch.no_grad():
        qLearningTargets = torch.zeros(globalVariables.miniBatchSize).to(device)

        rewards = sampledMiniBatch[2]
        nextStates = sampledMiniBatch[3]
        isNotTerminalStates = (~sampledMiniBatch[4])

        if globalVariables.deepQNetwork1Frozen:
            nextFrozenQValues = globalVariables.deepQNetwork1(nextStates)
        else:
            nextFrozenQValues = globalVariables.deepQNetwork2(nextStates)

        nextMaxFrozenQValues, _ = torch.max(nextFrozenQValues, 1)
        nextMaxFrozenQValues = nextMaxFrozenQValues.to(device)

        # If the state is a terminal state then there is no next state from which to get the estimated Q-value
        qLearningTargets = rewards + isNotTerminalStates*discountFactor*nextMaxFrozenQValues

    return qLearningTargets

def zeroOptimizerGrad():
    if globalVariables.deepQNetwork1Frozen:
        optimizer2.zero_grad()
    else:
        optimizer1.zero_grad()


def forwardProp(inputs):
    if globalVariables.deepQNetwork1Frozen:
        outputs = globalVariables.deepQNetwork2(inputs)
    else:
        outputs = globalVariables.deepQNetwork1(inputs)

    return outputs

def stepOptimizer():
    if globalVariables.deepQNetwork1Frozen:
        optimizer2.step()
    else:
        optimizer1.step()

def train():

    sampledMiniBatch = replayMemory.sampleMiniBatch(globalVariables.miniBatchSize)

    # Calculate the Q-learning targets
    qLearningTargets = calculateQLearningTargets(sampledMiniBatch)

    states = sampledMiniBatch[0]
    actions = sampledMiniBatch[1]

    zeroOptimizerGrad()

    qValuesAll = forwardProp(states)

    qValues = torch.zeros(globalVariables.miniBatchSize).to(device)
    for i in range(globalVariables.miniBatchSize):
        action = actions[i]
        qValues[i] = qValuesAll[i, action]

    

    loss = lossFunction(qValues, qLearningTargets)
    loss.backward()

    stepOptimizer()



def plotStatistics(fig, ax, line1, qValueRollingAverage, ax2, line2):

    line1.set_xdata(np.append(line1.get_xdata(), len(globalVariables.loggedAverageQValues)-1))
    line1.set_ydata(globalVariables.loggedAverageQValues)
    ax.relim()
    ax.autoscale_view()

    line2.set_xdata(np.append(line2.get_xdata(), len(globalVariables.loggedAverageQValues)-1))
    line2.set_ydata(np.append(line2.get_ydata(), qValueRollingAverage))
    ax2.relim()
    ax2.autoscale_view()

    fig.canvas.draw()
    


discountFactor = 0.95
startingEpsilon = 1
endingEpsilon = 0.1
stepEpsilon = (startingEpsilon - endingEpsilon)/100000 #1000000
oldState = None
qValueRollingAverage = None
replayMemory = ReplayMemory(globalVariables.replayMemorySize, device)

# Initialize statistics plot
plt.ion()
fig, (ax, ax2) = plt.subplots(2, 1)
line1, = ax.plot([], globalVariables.loggedScores)
line2, = ax2.plot([], [])
fig.show()
fig.canvas.draw()


globalVariables.epsilon = startingEpsilon

while running:

    screen = rendering.render(visualize)

    #time.sleep(0.01)

    logic.update()
    
    # Update the graphics of the window
    globalVariables.window.update()


# Game Loop
while running:

    '''
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            
            if globalVariables.deepQNetwork1Frozen:
                torch.save(globalVariables.deepQNetwork2.state_dict(), 'C:/Users/Rickard/Documents/python/MachineLearning/PythonSnakeDeepQ/savedNetworks/network6times4.pt')
            else:
                torch.save(globalVariables.deepQNetwork1.state_dict(), 'C:/Users/Rickard/Documents/python/MachineLearning/PythonSnakeDeepQ/savedNetworks/network6times4.pt')

            sys.exit()

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

    '''
    

    action, qValues = epsilonGreedy(stackedFrames, globalVariables.epsilon)
    globalVariables.qBuffer.append(qValues.cpu().detach().numpy())

    reward, screen, isTerminalState = step(action)

    # Preprocess the frame
    #resizedScreen = preprocessInput(screen)

    # Add the new frame to the stacked frames
    stackedFrames = updateState(stackedFrames, screen)

    # Store transition in replay memory
    if oldState is not None:
        replayMemory.addTransition(oldState, action, reward, stackedFrames, isTerminalState)
    oldState = stackedFrames

    if len(replayMemory.memory) >= replayMemory.memorySize:
        train()

    if globalVariables.numberOfSteps % globalVariables.numberStepsSwitchQNetwork == 0:

        # Switch the networks
        if globalVariables.deepQNetwork1Frozen:
            globalVariables.deepQNetwork1.load_state_dict(globalVariables.deepQNetwork2.state_dict())
        else:
            globalVariables.deepQNetwork2.load_state_dict(globalVariables.deepQNetwork1.state_dict())
        globalVariables.deepQNetwork1Frozen = (not globalVariables.deepQNetwork1Frozen)



    # If the state is a terminal state then the episode has ended, plot score of episode
    if isTerminalState:
        print(globalVariables.score, globalVariables.epsilon)
        globalVariables.loggedScores.append(globalVariables.score)
        qAverage = np.mean(np.array(globalVariables.qBuffer))
        globalVariables.loggedAverageQValues.append(qAverage)
        globalVariables.score = 0
        globalVariables.qBuffer = []
        stackedFrames = initializeState()

        if qValueRollingAverage is not None:
            qValueRollingAverage = qValueRollingAverage + 0.05*(qAverage - qValueRollingAverage)
        else:
            qValueRollingAverage = qAverage

        plotStatistics(fig, ax, line1, qValueRollingAverage, ax2, line2)

    #plotFrame(stackedFrames)
    #plt.pause(1)


    #if globalVariables.epsilon == 0: # or globalVariables.numberOfEpisodes >= 40000:
    #    clock.tick(10)

    #clock.tick(globalVariables.fps) # Use for rendering and showing the game
    #clock.tick() # Don't delay framerate when not rendering

    # Update the graphics of the window
    globalVariables.window.update()


if globalVariables.deepQNetwork1Frozen:
    torch.save(globalVariables.deepQNetwork2.state_dict(), 'C:/Users/Rickard/Documents/python/MachineLearning/PythonSnakeDeepQ/savedNetworks/network6times4.pt')
else:
    torch.save(globalVariables.deepQNetwork1.state_dict(), 'C:/Users/Rickard/Documents/python/MachineLearning/PythonSnakeDeepQ/savedNetworks/network6times4.pt')

pygame.quit()
