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
![](demoFinal.gif | width=100)
