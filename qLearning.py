import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import globalVariables

class ReplayMemory:
    def __init__(self, memorySize, device):
        self.memorySize = memorySize
        self.device = device
        
        self.states = torch.empty((memorySize, 4, globalVariables.downSampleWidth, globalVariables.downSampleHeight)).to(self.device)
        self.actions = torch.empty(memorySize, dtype=torch.int8).to(self.device)
        self.rewards = torch.empty(memorySize).to(self.device)
        self.nextStates = torch.empty((memorySize, 4, globalVariables.downSampleWidth, globalVariables.downSampleHeight)).to(self.device)
        self.isTerminalStates = torch.empty(memorySize, dtype=torch.bool).to(self.device)
        
        self.currentSize = 0

    def addTransition(self, state, action, reward, nextState, isTerminalState):
        state = state.to(self.device)
        action = torch.tensor([action], dtype=torch.int8).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        nextState = nextState.to(self.device)
        isTerminalState = torch.tensor([isTerminalState], dtype=torch.bool).to(self.device)

        if self.currentSize >= self.memorySize:
            self.states = torch.roll(self.states, -1, dims=0)
            self.actions = torch.roll(self.actions, -1, dims=0)
            self.rewards = torch.roll(self.rewards, -1, dims=0)
            self.nextStates = torch.roll(self.nextStates, -1, dims=0)
            self.isTerminalStates = torch.roll(self.isTerminalStates, -1, dims=0)

            self.states[-1,:,:,:] = state
            self.actions[-1] = action
            self.rewards[-1] = reward
            self.nextStates[-1,:,:,:] = nextState
            self.isTerminalStates[-1] = isTerminalState
        else:
            self.states[self.currentSize,:,:,:] = state
            self.actions[self.currentSize] = action
            self.rewards[self.currentSize] = reward
            self.nextStates[self.currentSize,:,:,:] = nextState
            self.isTerminalStates[self.currentSize] = isTerminalState

            self.currentSize = self.currentSize + 1


    def sampleMiniBatch(self, miniBatchSize):
        sampledIndices = np.random.choice(self.actions.shape[0], miniBatchSize, replace=False)

        states = self.states[sampledIndices,:,:,:]
        actions = self.actions[sampledIndices]
        rewards = self.rewards[sampledIndices]
        nextStates = self.nextStates[sampledIndices,:,:,:]
        isTerminalStates = self.isTerminalStates[sampledIndices]

        return (states,
                actions,
                rewards,
                nextStates,
                isTerminalStates)


class QLearning:
    def __init__(self, network, discountFactor=0.95, learningRate=1e-4, miniBatchSize=32, numberStepsSwitchQNetwork=2000, 
    startingEpsilon=1, endingEpsilon=0.1, numberStepsDecreasingEpsilon=1e6, downSampleWidth=116, downSampleHeight=94, 
    device=torch.device('cpu'), pretrainedPath=None):
        self.deepQNetwork1 = network()
        self.deepQNetwork2 = network()
        self.device = device
        self.deepQNetwork1.to(device)
        self.deepQNetwork2.to(device)
        self.learningRate = learningRate
        self.miniBatchSize = miniBatchSize

        self.downSampleWidth = downSampleWidth
        self.downSampleHeight = downSampleHeight

        self.deepQNetwork1Frozen = False
        self.numberStepsSwitchQNetwork = numberStepsSwitchQNetwork

        self.discountFactor = 0.95

        # Assign the initial and ending value to epsilon
        self.epsilon = startingEpsilon
        self.endingEpsilon = endingEpsilon

        # Calculate the change for epsilon in each step
        self.epsilonStep = (startingEpsilon - endingEpsilon)/numberStepsDecreasingEpsilon

        # Use a pretrained network if given a path to one
        if pretrainedPath is not None:
            self.deepQNetwork1.load_state_dict(torch.load(pretrainedPath))

        # Set the starting parameters to be the same for both networks
        self.deepQNetwork2.load_state_dict(self.deepQNetwork1.state_dict())

        # Set the optimizer
        self.optimizer1 = optim.SGD(self.deepQNetwork1.parameters(), lr=learningRate, momentum=0.9)
        self.optimizer2 = optim.SGD(self.deepQNetwork2.parameters(), lr=learningRate, momentum=0.9)

        # Set the loss function
        self.lossFunction = nn.MSELoss()



    def preprocessInput(self, frame):

        # Convert to gray-scale
        frame = 0.2989*frame[:, :, 0] + 0.5870*frame[:, :, 1] + 0.1140*frame[:, :, 2]

        # Add dimensions and convert to float
        frame = frame.permute(1, 0).unsqueeze(0).unsqueeze(1).float()

        # Down-sample the image
        frame = F.interpolate(frame, size=(self.downSampleWidth, self.downSampleHeight), mode = 'bilinear')

        return frame

    def initializeState(self, screen):

        # Preprocess teh current frame (Convert to grayscale and resize the frame to fit the newtork input)
        resizedScreen = self.preprocessInput(screen)

        # Initialize the state by adding the current frame to three empty frames
        stackedFrames = torch.tensor(np.zeros((1, 3, 116, 94)), dtype=torch.float32)
        stackedFrames = torch.cat((resizedScreen.detach().clone(), stackedFrames), 1)

        return stackedFrames

    def updateState(self, stackedFrames, frame):

        # Shift the frames in the state
        stackedFrames = torch.roll(stackedFrames, 1, dims=1)

        # Replace the oldest frame with the new frame
        stackedFrames[:,0,:,:] = frame

        return stackedFrames

    def greedy(self, state):

        state = state.to(self.device)
        with torch.no_grad():
            if self.deepQNetwork1Frozen:
                qValues = self.deepQNetwork2(state)
            else:
                qValues = self.deepQNetwork1(state)
            action = np.argmax(qValues.cpu().detach().numpy())

        return action, qValues

    def epsilonGreedy(self, state):

        assert(self.epsilon <= 1)

        # Draw a random number from uniform distribution (0-1)
        random_value = np.random.uniform()

        # Determine the greedy action
        greedyAction, qValues = self.greedy(state)

        # With probability epsilon select a random action, otherwise select the greedy action
        if random_value < self.epsilon:
            action = np.random.randint(0, 4)
        else:
            action = greedyAction

        return action, qValues

    def calculateQLearningTargets(self, sampledMiniBatch):
        # Calculate the Q-learning targets for the given mini-batch

        with torch.no_grad():
            qLearningTargets = torch.zeros(self.miniBatchSize).to(self.device)

            # Extract the components of the mini-batch
            rewards = sampledMiniBatch[2]
            nextStates = sampledMiniBatch[3]
            isNotTerminalStates = (~sampledMiniBatch[4])

            if self.deepQNetwork1Frozen:
                nextFrozenQValues = self.deepQNetwork1(nextStates)
            else:
                nextFrozenQValues = self.deepQNetwork2(nextStates)

            # Use the estimated Q-value given that one takes the best action from the next 
            # state as an approximation for how much value the next state has
            nextMaxFrozenQValues, _ = torch.max(nextFrozenQValues, 1)
            nextMaxFrozenQValues = nextMaxFrozenQValues.to(self.device)

            # Calculate the Q-learning targets
            # If the state is a terminal state then there is no next state from which to get the estimated Q-value thus
            # isNotTerminalStates is used to multiply by 0 when isNotTerminalStates is false
            qLearningTargets = rewards + isNotTerminalStates*self.discountFactor*nextMaxFrozenQValues

        return qLearningTargets
    
    def switchQNetworks(self):
        # Switch the networks
        if self.deepQNetwork1Frozen:
            self.deepQNetwork1.load_state_dict(self.deepQNetwork2.state_dict())
        else:
            self.deepQNetwork2.load_state_dict(self.deepQNetwork1.state_dict())
        self.deepQNetwork1Frozen = (not self.deepQNetwork1Frozen)

    def zeroOptimizerGrad(self):
        if self.deepQNetwork1Frozen:
            self.optimizer2.zero_grad()
        else:
            self.optimizer1.zero_grad()


    def forwardProp(self, inputs):
        if self.deepQNetwork1Frozen:
            outputs = self.deepQNetwork2(inputs)
        else:
            outputs = self.deepQNetwork1(inputs)

        return outputs

    def stepOptimizer(self):
        if self.deepQNetwork1Frozen:
            self.optimizer2.step()
        else:
            self.optimizer1.step()

    def stepEpsilon(self):
        self.epsilon = max(self.epsilon - self.epsilonStep, self.endingEpsilon)
