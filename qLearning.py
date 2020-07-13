import torch
import numpy as np
from collections import deque
import globalVariables

class ReplayMemory:
    def __init__(self, memorySize, device):
        self.memorySize = memorySize
        self.memory = deque(maxlen = self.memorySize)
        self.device = device

    def addTransition(self, state, action, reward, nextState, isTerminalState):
        self.memory.append([state,
                            action,
                            reward,
                            nextState,
                            isTerminalState])

    def sampleMiniBatch(self, miniBatchSize):
        sampledIndices = np.random.choice(len(self.memory), miniBatchSize, replace=False)

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
