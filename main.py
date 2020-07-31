import sys
import pygame
import rendering
import logic
import snake
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

snake.initalizeGame(pygame)

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
    pretrainedPath = 'savedNetworks/network12times8Final3.pt'
else:
    pretrainedPath = None

# Parameter settings
discountFactor = 0.95
startingEpsilon = 0.01
endingEpsilon = 0.01
numberStepsDecreasingEpsilon = 1e5 #1000000
replayMemorySize = 1e4 #2500
numberStepsSwitchQNetwork = 512 #2500
miniBatchSize = 32
learningRate = 1e-4 #1e-5
downSampleWidth = 116
downSampleHeight = 94

# Initialize variables
oldState = None
qValueRollingAverage = None

# Initialize replay memory
replayMemory = ReplayMemory(replayMemorySize, device, fastImplementation=True)

# Define the Q-network and Q-learning parameters
qLearning = QLearning(DeepQNetwork, discountFactor, learningRate, miniBatchSize, 
    numberStepsSwitchQNetwork, startingEpsilon, endingEpsilon, numberStepsDecreasingEpsilon, 
    device=device, pretrainedPath=pretrainedPath)

# Get the current frame of the game and initialize the state
frame = snake.getGameFrame(pygame)
stackedFrames = qLearning.initializeState(frame)


'''
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
            if qLearning.deepQNetwork1Frozen:
                torch.save(qLearning.deepQNetwork2.state_dict(), 'savedNetworks/savedNetwork.pt')
            else:
                torch.save(qLearning.deepQNetwork1.state_dict(), 'savedNetworks/savedNetwork.pt')

            sys.exit()

        # Event-handling for keyboard input when a human plays the game
        logic.handleKeyboardInput(event)
    

    # Select the action from an epsilon-greedy policy
    action, qValues = qLearning.epsilonGreedy(stackedFrames)

    # Save the Q-values
    globalVariables.qMaxBuffer.append(qValues.cpu().detach().numpy().max())

    # Take a step with the selected action, recieve a reward, new scree-frame 
    # and if a terminal state has been reached (i.e. game over)
    reward, frame, isTerminalState = snake.step(action)

    # Update epsilon (decreases as to reduce the amount of exploration over time)
    qLearning.stepEpsilon()

    # Preprocess the frame
    resizedframe = qLearning.preprocessInput(frame)

    # Add the new frame to the stacked frames
    stackedFrames = qLearning.updateState(stackedFrames, resizedframe)

    # Store transition in replay memory
    if oldState is not None:
        replayMemory.addTransition(oldState, action, reward, stackedFrames, isTerminalState)
    oldState = stackedFrames

    # When the replay-memory is of sufficient size, start training
    if replayMemory.currentSize >= miniBatchSize: #replayMemory.currentSize >= replayMemory.memorySize:
        qLearning.train(replayMemory)

    # The "fixed Q-targets"-idea from the "Playing Atari with Deep Reinforcement Learning"-paper
    if globalVariables.numberOfSteps % qLearning.numberStepsSwitchQNetwork == 0:
        # Switch the networks
        qLearning.switchQNetworks()


    # If the state is a terminal state then the episode has ended, plot score of episode
    if isTerminalState:
        globalVariables.loggedScores.append(globalVariables.score)
        qMaxAverage = np.mean(np.array(globalVariables.qMaxBuffer))
        globalVariables.loggedAverageQValues.append(qMaxAverage)

        if qValueRollingAverage is not None:
            qValueRollingAverage = qValueRollingAverage + 0.05*(qMaxAverage - qValueRollingAverage)
        else:
            qValueRollingAverage = qMaxAverage

        print("Game over, Episode number:", globalVariables.numberOfEpisodes, '\t', globalVariables.score, 
            '\t', qLearning.epsilon, '\t', globalVariables.numberOfSteps, '\t', "Rolling average max Q-value:", qValueRollingAverage)
        #plotStatistics(fig, ax, line1, qValueRollingAverage, ax2, line2)

        globalVariables.score = 0
        globalVariables.qBuffer = []
        frame = snake.getGameFrame(pygame)
        stackedFrames = qLearning.initializeState(frame)

    if qLearning.epsilon == 0:
        clock.tick(10)

    #clock.tick(globalVariables.fps) # Use for rendering and showing the game
    clock.tick() # Don't delay framerate when not rendering


pygame.quit()
