import sys, pygame, random, math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import sgd
from qlearning import ExperienceReplay


#pygame.init()

size = width, height = 300, 300
speed = [1, 1]
BLACK = 0, 0, 0
RED =   (255,   0,   0)
YELLOW = (255,255,0)

mouseColor = RED
mouseDim = 20,20
mousePos = np.array([100,100])

cheeseColor = YELLOW
cheeseDim = 20,20
cheesePos = 0,100

moveTop = [0,-1]
moveBottom = [0,1]
moveLeft = [-1,0]
moveRight = [1,0]
actions = np.zeros((4,2))
actions[0] = moveTop
actions[1] = moveBottom
actions[2] = moveLeft
actions[3] = moveRight

Qmatrix = np.zeros((15,10,4))
learningRate = 0.5
exploRate = 1.0
discountRate = 0.95

Qmatrix[1][2][3] = 50

mousePos += moveTop

#screen = pygame.display.set_mode(size)

grid_size = 15
hidden_size = 100
nb_frames = 1

gameMap = np.zeros((nb_frames, grid_size, grid_size))
gameMap[0][cheesePos[0]/20][cheesePos[1]/20] = 1
gameMap[0][mousePos[0]/20][mousePos[1]/20] = 2

nn = Sequential()
nn.add(Flatten(input_shape=(nb_frames, grid_size, grid_size)))
nn.add(Dense(hidden_size, activation='relu'))
nn.add(Dense(hidden_size, activation='relu'))
nn.add(Dense(4))
nn.compile(sgd(lr=.2), "mse")

memory = ExperienceReplay(100)
loss = 0.

def nnEvalSate(pos, gameMap, exploRate, loss):
    mouseState = np.array(pos/20)
    nnoutputs =  nn.predict(np.array([gameMap]), batch_size=1)        #nn.feed_forward(mouseState)
    nnoutputs = nnoutputs[0]                                           #nnoutputs = np.array(nn.output_layer.get_outputs())
    actionId = 0
    if random.random() < exploRate:
        actionId = random.randrange(4)
    else:
        actionId = random.choice(np.argwhere(nnoutputs == nnoutputs.max()).flatten())

    Rpoints, newMouseState = reward(mouseState, actions[actionId])
    newGameMap = np.zeros((nb_frames, grid_size, grid_size))
    np.copyto(newGameMap, gameMap)
    newGameMap[0][mouseState[0]][mouseState[1]] = 0
    newGameMap[0][newMouseState[0]][newMouseState[1]] = 2
    memory.remember(np.array([gameMap]), actionId, Rpoints, np.array([newGameMap]), False)
    batch = memory.get_batch(model=nn, batch_size=5, gamma=0.9)
    if batch:
        inputs, targets = batch
        loss += float(nn.train_on_batch(inputs, targets))
    #print("Loss {:.4f}".format(loss))
    # Qstate = nnoutputs[actionId]
    # QmaxNextState = 0
    # if not np.array_equal(mouseState[0], newMouseState):
    #     QmaxNextState = nn.predict(np.array([newGameMap])).max()    #nn.feed_forward(newMouseState)
    #         #QmaxNextState = np.array(nn.output_layer.get_outputs()).max()
    # Qtargets = nnoutputs
    # Qtargets[actionId] = Rpoints + discountRate * QmaxNextState

    #target_error = float(nn.train_on_batch(np.array([gameMap]), np.array([Qtargets])))    #target_error = nn.train(mouseState, Qtargets)
    gameMap = newGameMap

    return newMouseState*20, loss

def evalSate(pos):
    mouseState = pos/20
    state = Qmatrix[mouseState[0]][mouseState[1]]

    actionId = 0
    if random.random() < exploRate:
        actionId = random.randrange(4)
    else:
        actionId = random.choice(np.argwhere(state == state.max()).flatten())

    Rpoints, newMouseState = reward(mouseState, actions[actionId])
    Qstate = state[actionId]
    QmaxNextState = 0
    if not np.array_equal(mouseState, newMouseState):
        QmaxNextState = QmaxForNextState(newMouseState)
    newQstate = Qstate + learningRate * (Rpoints + discountRate * QmaxNextState - Qstate)

    Qmatrix[mouseState[0]][mouseState[1]][actionId] = newQstate
    return newMouseState*20


def reward(state, action):
    result = -10
    resultState = state
    target = np.array(cheesePos)
    target /= 20
    newState = state + action
    if newState[0] >= 0 and newState[0] < width/20:
        if newState[1] >= 0 and newState[1] < height/20:
            distance = math.fabs(newState[0] - target[0]) + math.fabs(newState[1] - target[1])
            if distance == 0:
                result = 100
            else:
                result = -1
            resultState = newState

    return result, resultState

def QmaxForNextState(state):
    return Qmatrix[state[0]][state[1]].max()

steps = 0
while 1:
    #pygame.time.wait(10)
    #for event in pygame.event.get():
        #if event.type == pygame.QUIT: sys.exit()

    mousePos, target_error = nnEvalSate(mousePos, gameMap, exploRate, loss)
    loss += target_error
    if exploRate > 0.1:
        exploRate -= 0.000001
    steps += 1
    if np.array_equal(mousePos, cheesePos):
        mousePos = np.array([100,100])
        print steps, exploRate, loss
        steps = 0

    #screen.fill(BLACK)
    #pygame.draw.rect(screen, cheeseColor, [cheesePos,cheeseDim])
    #pygame.draw.rect(screen, mouseColor, [mousePos,mouseDim])
    #pygame.display.flip()
