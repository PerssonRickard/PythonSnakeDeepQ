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
#from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from collections import deque
from qLearning import ReplayMemory, QLearning

pygame.init()

clock = pygame.time.Clock()

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

globalVariables.screen = pygame.display.set_mode(globalVariables.size)
logic.updateApplePos()
pygame.display.set_caption('Snake')

globalVariables.snake_list = logic.createSnakeList()

# Render the initial frame
rendering.render()

def plotFrame(frame):
    # Used for debugging, plots the given frame

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



class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

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


# Load a pretrained network
if globalVariables.pretrained:
    pretrainedPath = 'C:/Users/Rickard/Documents/python/MachineLearning/PythonSnakeDeepQ/savedNetworks/network12times8FinalNewRewards.pt'


# Parameter settings
discountFactor = 0.95
startingEpsilon = 0.1
endingEpsilon = 0.05
numberStepsDecreasingEpsilon = 1e5 #1000000
oldState = None
qValueRollingAverage = None
replayMemory = ReplayMemory(globalVariables.replayMemorySize, device)

# Define the Q-network
QLearning = QLearning(DeepQNetwork, discountFactor, globalVariables.learningRate, globalVariables.miniBatchSize, 
    globalVariables.numberStepsSwitchQNetwork, startingEpsilon, endingEpsilon, numberStepsDecreasingEpsilon, globalVariables.downSampleWidth, 
    globalVariables.downSampleHeight, device=device, pretrainedPath=pretrainedPath)


# Get the current frame of the game
window_pixel_matrix = pygame.surfarray.pixels3d(globalVariables.screen)
screen = torch.from_numpy(np.copy(window_pixel_matrix))
stackedFrames = QLearning.initializeState(screen)


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

    # Update epsilon (decreases as to reduce the amount of exploration over time)
    QLearning.stepEpsilon()

    return reward, screen, isTerminalState

def train():

    # Sample a mini-batch of experience
    sampledMiniBatch = replayMemory.sampleMiniBatch(globalVariables.miniBatchSize)

    # Calculate the Q-learning targets
    qLearningTargets = QLearning.calculateQLearningTargets(sampledMiniBatch)

    # Extract components of the mini-batch
    states = sampledMiniBatch[0]
    actions = sampledMiniBatch[1]

    # Zero the optimizer gradients
    QLearning.zeroOptimizerGrad()

    # Perform a forward-pass of the Q-network
    qValuesAll = QLearning.forwardProp(states)

    # Only keep the Q-values corresponding to actions that we actually did take
    qValues = torch.zeros(globalVariables.miniBatchSize).to(device)
    for i in range(globalVariables.miniBatchSize):
        action = actions[i]
        qValues[i] = qValuesAll[i, action]

    # Compute the loss
    loss = QLearning.lossFunction(qValues, qLearningTargets)

    # Perform backpropagation to obtain the gradients
    loss.backward()

    # Perform an update step of the Q-network
    QLearning.stepOptimizer()



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
    

'''
# Initialize statistics plot
plt.ion()
fig, (ax, ax2) = plt.subplots(2, 1)
line1, = ax.plot([], globalVariables.loggedScores)
line2, = ax2.plot([], [])
fig.show()
fig.canvas.draw()
'''

# Game Loop
while 1:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            
            # When the application is closed save the Q-network
            if QLearning.deepQNetwork1Frozen:
                torch.save(QLearning.deepQNetwork2.state_dict(), 'C:/Users/Rickard/Documents/python/MachineLearning/PythonSnakeDeepQ/savedNetworks/network6times4.pt')
            else:
                torch.save(QLearning.deepQNetwork1.state_dict(), 'C:/Users/Rickard/Documents/python/MachineLearning/PythonSnakeDeepQ/savedNetworks/network6times4.pt')

            sys.exit()

        # Event-handling for keyboard input when a human plays the game
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
    

    # Select the action from an epsilon-greedy policy
    action, qValues = QLearning.epsilonGreedy(stackedFrames)

    # Save the Q-values
    globalVariables.qMaxBuffer.append(qValues.cpu().detach().numpy().max())

    # Take a step with the selected action, recieve a reward, new scree-frame 
    # and if a terminal state has been reached (i.e. game over)
    reward, screen, isTerminalState = step(action)

    # Preprocess the frame
    resizedScreen = QLearning.preprocessInput(screen)

    # Add the new frame to the stacked frames
    stackedFrames = QLearning.updateState(stackedFrames, resizedScreen)

    # Store transition in replay memory
    if oldState is not None:
        replayMemory.addTransition(oldState, action, reward, stackedFrames, isTerminalState)
    oldState = stackedFrames

    # When the replay-memory has been filled, start training
    if replayMemory.currentSize >= replayMemory.memorySize:
        train()

    # The "fixed Q-targets"-idea from the "Playing Atari with Deep Reinforcement Learning"-paper
    if globalVariables.numberOfSteps % QLearning.numberStepsSwitchQNetwork == 0:
        # Switch the networks
        QLearning.switchQNetworks()


    # If the state is a terminal state then the episode has ended, plot score of episode
    if isTerminalState:
        globalVariables.loggedScores.append(globalVariables.score)
        qMaxAverage = np.mean(np.array(globalVariables.qMaxBuffer))
        globalVariables.loggedAverageQValues.append(qMaxAverage)

        if qValueRollingAverage is not None:
            qValueRollingAverage = qValueRollingAverage + 0.05*(qMaxAverage - qValueRollingAverage)
        else:
            qValueRollingAverage = qMaxAverage

        print(globalVariables.score, QLearning.epsilon, globalVariables.numberOfSteps, "Rolling average max Q-value:", qValueRollingAverage)
        #plotStatistics(fig, ax, line1, qValueRollingAverage, ax2, line2)

        globalVariables.score = 0
        globalVariables.qBuffer = []
        window_pixel_matrix = pygame.surfarray.pixels3d(globalVariables.screen)
        screen = torch.from_numpy(np.copy(window_pixel_matrix))
        stackedFrames = QLearning.initializeState(screen)

    #plotFrame(stackedFrames)
    #plt.pause(1)


    if QLearning.epsilon == 0: # or globalVariables.numberOfEpisodes >= 40000:
        clock.tick(10)

    #clock.tick(globalVariables.fps) # Use for rendering and showing the game
    clock.tick() # Don't delay framerate when not rendering


pygame.quit()
