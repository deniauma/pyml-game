import sys, pygame, random
import numpy as np
from game import Explo1
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import RMSprop
from qlearning import ExperienceReplay


pixelsPerBlock = 10
grid_size = 10
nb_frames = 1
nb_actions = 3
explo = 0.3

live = False
if live:
    pygame.init()
    screen = pygame.display.set_mode((grid_size * pixelsPerBlock, grid_size * pixelsPerBlock))

BLACK = 0, 0, 0
FLOOR = (100,100,100)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
RAY = (0, 255, 0)

model = Sequential()
model.add(Convolution2D(16, nb_row=3, nb_col=3, activation='relu', input_shape=(nb_frames, 3 * pixelsPerBlock, 3 * pixelsPerBlock)))
model.add(Convolution2D(32, nb_row=3, nb_col=3, activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(nb_actions))
model.compile(RMSprop(), 'MSE')

memory = ExperienceReplay(100)
total_loss = 0.

displayGrid = True

game = Explo1(grid_size, pixelsPerBlock)


def drawGrid(grid_size, prixelsPerBlock):
    for i in range(grid_size):
        startH = np.array((0, pixelsPerBlock)) * i
        stopH = np.array((grid_size * pixelsPerBlock, pixelsPerBlock)) * i
        startV = np.array((pixelsPerBlock, 0)) * i
        stopV = np.array((pixelsPerBlock, pixelsPerBlock * pixelsPerBlock)) * i
        pygame.draw.line(screen, WHITE, startH, stopH)
        pygame.draw.line(screen, WHITE, startV, stopV)


def drawMap(surface, map):
    surface.fill(FLOOR)
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j] == 1:
                start = np.array((i, j)) * pixelsPerBlock
                dim = [pixelsPerBlock, pixelsPerBlock]
                pygame.draw.rect(surface, YELLOW, [start, dim])

def drawRay(surface, ray):
    rrect = pygame.draw.polygon(surface, BLACK, ray, 0)
    # pygame.draw.rect(screen, (0,0,255), rrect)
    return rrect

def rgb2gray(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def preprocess_frame(surface, frame_rect):
    if frame_rect[0] + frame_rect[2] > grid_size * pixelsPerBlock:
        frame_rect[2] = grid_size * pixelsPerBlock - frame_rect[0]
    if frame_rect[1] + frame_rect[3] > grid_size * pixelsPerBlock:
        frame_rect[3] = grid_size * pixelsPerBlock - frame_rect[1]
    robot_vision = surface.subsurface(frame_rect)
    extract_ray = pygame.Surface((3 * pixelsPerBlock, 3 * pixelsPerBlock))
    extract_ray.fill(BLACK)
    extract_ray.blit(robot_vision, (0, 0))
    # screen.blit(extract_ray, (0, 0))
    robot_vision_pixels = pygame.surfarray.array2d(extract_ray)
    # frame_greyscale = np.zeros((3 * pixelsPerBlock, 3 * pixelsPerBlock))
    # for i in range(3 * pixelsPerBlock):
    #     for j in range(3 * pixelsPerBlock):
    #         frame_greyscale[i][j] = rgb2gray(robot_vision_pixels[i][j])
    return robot_vision_pixels

def get_game_state(game, for_display=False):
    robotPos, robotDir, state, ray_vision = game.getGameData()
    state_map = pygame.Surface((grid_size * pixelsPerBlock, grid_size * pixelsPerBlock))
    drawMap(state_map, state)

    # Draw robot
    pygame.draw.circle(state_map, RED, robotPos + pixelsPerBlock / 2, pixelsPerBlock / 2)
    startDir = robotPos + pixelsPerBlock / 2
    stopDir = startDir + robotDir * pixelsPerBlock
    pygame.draw.line(state_map, RED, startDir, stopDir)

    # Compute light map
    light = pygame.Surface((grid_size * pixelsPerBlock, grid_size * pixelsPerBlock))
    light.fill(pygame.color.Color('Grey'))
    ray_rect = drawRay(light, ray_vision)

    # Get current game state
    game_state = preprocess_frame(state_map, ray_rect)

    if for_display:
        # Apply lightning
        state_map.blit(light, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
    return game_state, state_map

def play_game(game, explo_rate):
    #Get current game data
    game_state, entire_map = get_game_state(game, True)
    game_state = np.expand_dims(game_state, 0)
    nnoutputs = model.predict(np.array([game_state]), batch_size=1)  # nn.feed_forward(mouseState)
    nnoutputs = nnoutputs[0]

    #Choose next game action
    action = 0
    if random.random() < explo_rate:
        action = random.randrange(nb_actions)
    else:
        action = random.choice(np.argwhere(nnoutputs == nnoutputs.max()).flatten())

    #Play choosen action and get new game state
    reward = game.play(action)  # random.randrange(3)
    print action, reward
    new_game_state, new_surface = get_game_state(game)

    #Compute transition
    memory.remember(np.array([game_state]), action, reward, np.array([new_game_state]), False)
    batch = memory.get_batch(model=model, batch_size=5, gamma=0.9)
    if batch:
        inputs, targets = batch
        loss = float(model.train_on_batch(inputs, targets))

    return entire_map

while 1:
    pygame.time.wait(100)
    if live:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP: game.play(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT: game.play(1)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT: game.play(2)

    game_surface = play_game(game, explo)

    if live:
        screen.blit(game_surface, (0, 0))
        if displayGrid:
            drawGrid(grid_size, pixelsPerBlock)

        pygame.display.flip()

