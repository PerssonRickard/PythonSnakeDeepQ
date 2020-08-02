# PythonSnakeDeepQ

This project consists of training an agent on the classical game of Snake using deep reinforcement learning. More specifically a version of Q-learning is used which was proposed in the paper ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602).

## Installation
The installation steps will be presented in the context of a Windows operating system but the requirements to run the project may also be available for other operating systems. 

### Anaconda
A simple and quick way to install the requirements for the project is to make use of the environment.yml file included in the repository. To use this file you need to install Anaconda which is available for Windows, MacOS and Linux at: https://www.anaconda.com/.

### Creating an environment with the required libraries
Once you have installed Anaconda you can create an environment which includes the required libraries by running the following command:
```
conda env create -f environment.yml
```

### Activating the created environment
```
conda activate snakeDeepQ
```

### Running the code
When you have activated the environment you can run the code by typing:
```
python main.py
```

## Demo
Initally the agent chooses actions based on an ε-greedy policy which means that it takes a random action with probability ε and otherwise the greedy action. This is to ensure that the agent explores the environment sufficiently as to not end up using a sub-optimal policy. In the image below it can be seen how the agent essentially chooses actions completely random at the start of the training process.

<p align="center">
  <img src="demoStart.gif" width="400"/>
</p>

After training for a sufficiently long time the agent learns to avoid the edges and to move towards the objective which can be seen in the image below.

<p align="center">
  <img src="demoFinal.gif" width="400"/>
</p>

## Conclusion
Using the version of Q-learning used in this project the agent is able to demonstrate intelligent behaviour in the sense that it is able to learn that it should avoid the edges and move towards the objective (the red square). The agent is able to achieve a substantial score but it seems to have a hard time learning to not close itself into situations where it is impossible to surivive, i.e., where the snake blocks itself with its own body. By providing the agent with more training time or by constructing some hand-engineered reward function it may be possible for the agent to obtain an even better score.
