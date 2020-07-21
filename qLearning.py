import torch
import numpy as np
from collections import deque
import globalVariables

class ReplayMemory:
    def __init__(self, memorySize, device):
        self.memorySize = memorySize
        #self.memory = deque(maxlen = self.memorySize)
        self.device = device
        self.memory = []

        # May be worth trying to improve performance of sampleMiniBatch
        #self.states = torch.empty((0, 4, globalVariables.downSampleWidth, globalVariables.downSampleHeight)).to(self.device)
        #self.actions = torch.empty(0, dtype=torch.int8).to(self.device)
        #self.rewards = torch.empty(0).to(self.device)
        #self.nextStates = torch.empty((0, 4, globalVariables.downSampleWidth, globalVariables.downSampleHeight)).to(self.device)
        #self.isTerminalStates = torch.empty(0, dtype=torch.bool).to(self.device)

    def addTransition(self, state, action, reward, nextState, isTerminalState):

        if len(self.memory) >= self.memorySize:
            temp = self.memory[1:]
        else:
            temp = self.memory

        temp.append([state,
                    action,
                    reward,
                    nextState,
                    isTerminalState])

        self.memory = temp

    def sampleMiniBatch(self, miniBatchSize):
        sampledIndices = np.random.choice(self.memorySize, miniBatchSize, replace=False)

        states = torch.zeros((miniBatchSize, 4, globalVariables.downSampleWidth, globalVariables.downSampleHeight)).to(self.device)
        actions = torch.zeros(miniBatchSize, dtype=torch.int8).to(self.device)
        rewards = torch.zeros(miniBatchSize).to(self.device)
        nextStates = torch.zeros((miniBatchSize, 4, globalVariables.downSampleWidth, globalVariables.downSampleHeight)).to(self.device)
        isTerminalStates = torch.zeros(miniBatchSize, dtype=torch.bool).to(self.device)

        for i in range(miniBatchSize):
            memoryIndex = sampledIndices[i]

            states[i,:,:,:] = self.memory[memoryIndex][0]
            actions[i] = self.memory[memoryIndex][1]
            rewards[i] = self.memory[memoryIndex][2]
            nextStates[i,:,:,:] = self.memory[memoryIndex][3]
            isTerminalStates[i] = self.memory[memoryIndex][4]

        return (states,
                actions,
                rewards,
                nextStates,
                isTerminalStates)
