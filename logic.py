import random
import globalVariables


def update():
    globalVariables.snake_direction = globalVariables.pending_snake_direction

    isGameOver = checkIfGameOver()

    checkIfAteApple()
    updateSnakeList()

    if isGameOver:
        print("Game over, ", globalVariables.numberOfIterations)
        globalVariables.numberOfIterations = globalVariables.numberOfIterations + 1
        resetGame()


def updateSnakeList():

    snake_head_pos = globalVariables.snake_list[0]
    temp_pos1 = snake_head_pos[:]

    if globalVariables.snake_direction == 0:
        snake_head_pos[1] = snake_head_pos[1] - 1
    elif  globalVariables.snake_direction == 1:
        snake_head_pos[0] = snake_head_pos[0] + 1
    elif  globalVariables.snake_direction == 2:
        snake_head_pos[1] = snake_head_pos[1] + 1
    elif  globalVariables.snake_direction == 3:
        snake_head_pos[0] = snake_head_pos[0] + -1

    count = 0
    for i, snake_body_pos in enumerate(globalVariables.snake_list[1:]):
        if i%2 == 0:
            temp_pos2 = snake_body_pos[:]
            globalVariables.snake_list[i+1] = temp_pos1[:]
        else:
            temp_pos1 = snake_body_pos[:]
            globalVariables.snake_list[i+1] = temp_pos2[:]
        count = i

    if len(globalVariables.snake_list) < globalVariables.snake_length: # If the if-statement is true then the apple has been eaten
        if count%2 == 0:
            globalVariables.snake_list.append(temp_pos2[:])
        else:
            globalVariables.snake_list.append(temp_pos1[:])


def updateApplePos():

    combinations = [[x, y] for x in range(globalVariables.grid_width) for y in range(globalVariables.grid_height-1)]
    for position in globalVariables.snake_list:
        if position in combinations:
            combinations.remove(position)

    pos = random.choice(combinations)
    
    globalVariables.apple_pos_x, globalVariables.apple_pos_y =  pos[0], pos[1]

def checkIfAteApple():
    snake_head_pos = globalVariables.snake_list[0][:]

    if globalVariables.snake_direction == 0:
        snake_head_pos[1] = snake_head_pos[1] - 1
    elif  globalVariables.snake_direction == 1:
        snake_head_pos[0] = snake_head_pos[0] + 1
    elif  globalVariables.snake_direction == 2:
        snake_head_pos[1] = snake_head_pos[1] + 1
    elif  globalVariables.snake_direction == 3:
        snake_head_pos[0] = snake_head_pos[0] + -1
    
    if snake_head_pos == [globalVariables.apple_pos_x, globalVariables.apple_pos_y]:
        globalVariables.snake_length += 1
        updateApplePos()

def checkIfGameOver():
    isGameOver = False

    snake_head_pos = globalVariables.snake_list[0][:]

    if globalVariables.snake_direction == 0:
        snake_head_pos[1] = snake_head_pos[1] - 1
    elif  globalVariables.snake_direction == 1:
        snake_head_pos[0] = snake_head_pos[0] + 1
    elif  globalVariables.snake_direction == 2:
        snake_head_pos[1] = snake_head_pos[1] + 1
    elif  globalVariables.snake_direction == 3:
        snake_head_pos[0] = snake_head_pos[0] + -1

    if snake_head_pos[0] < 0 or snake_head_pos[0] >= globalVariables.grid_width:
        isGameOver = True
    if snake_head_pos[1] < 0 or snake_head_pos[1] >= globalVariables.grid_height:
        isGameOver = True
    for snake_body_pos in globalVariables.snake_list[1:]:
        if snake_head_pos == snake_body_pos:
            isGameOver = True

    return isGameOver

def resetGame():
    globalVariables.snake_list = createSnakeList()
    while(not checkIfSnakeListFeasible(globalVariables.snake_list)):
        globalVariables.snake_list = createSnakeList()

    globalVariables.snake_length = globalVariables.initial_snake_length
    updateApplePos()


def createSnakeList():

    # NEED TO IMPLEMENT MULTIPLE POSSIBLE STARTING DIRECTIONS
    initial_snake_head_pos_x = random.randint(0, globalVariables.grid_width-1)
    initial_snake_head_pos_y = random.randint(0, globalVariables.grid_height-1)

    globalVariables.initial_snake_direction = random.randint(0, 3)
    globalVariables.pending_snake_direction = globalVariables.initial_snake_direction

    if globalVariables.initial_snake_direction == 0:
        snake_list = [[initial_snake_head_pos_x, initial_snake_head_pos_y + i] for i in range(globalVariables.initial_snake_length)]
    elif globalVariables.initial_snake_direction == 1:
        snake_list = [[initial_snake_head_pos_x - i, initial_snake_head_pos_y] for i in range(globalVariables.initial_snake_length)]
    elif globalVariables.initial_snake_direction == 2:
        snake_list = [[initial_snake_head_pos_x, initial_snake_head_pos_y - i] for i in range(globalVariables.initial_snake_length)]
    elif globalVariables.initial_snake_direction == 3:
        snake_list = [[initial_snake_head_pos_x + i, initial_snake_head_pos_y] for i in range(globalVariables.initial_snake_length)]

    return snake_list

def checkIfSnakeListFeasible(snake_list):
    isFeasible = True
    for snake_pos in snake_list:
        if snake_pos[0] < 0 or snake_pos[0] >= globalVariables.grid_width:
            isFeasible = False
        if snake_pos[1] < 0 or snake_pos[1] >= globalVariables.grid_height:
            isFeasible = False

    return isFeasible



