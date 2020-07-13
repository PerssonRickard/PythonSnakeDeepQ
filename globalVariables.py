fps = 30
downSampleWidth = 116
downSampleHeight = 94
grid_size = grid_width, grid_height = 12, 8 #6, 4  #32, 26
numberOfEpisodes = 0

snake_block_size = 25
initial_snake_length = 4
snake_direction = None # 0 = north, 1 = east, 2 = south, 3 = west
snake_speed = 55 # 0-60

size = width, height = grid_width*snake_block_size, grid_height*snake_block_size
screen = None

step_size = snake_block_size
snake_length = initial_snake_length
pending_snake_direction = snake_direction


snake_list = []
apple_pos_x, apple_pos_y = None, None
score = 0
loggedScores = []
loggedAverageQValues = []
qBuffer = []

deepQNetwork1 = None
deepQNetwork2 = None
replayMemorySize = 2500
deepQNetwork1Frozen = False
epsilon = 1
numberOfEpochs = 2
numberStepsSwitchQNetwork = 512 #2500
numberOfSteps = 0
miniBatchSize = 32
learningRate = 1e-4 #1e-5

line1 = []
fig = None
ax = None

pretrained = True
