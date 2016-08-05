import sys, pygame, random
import numpy as np
from game import Explo1
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import sgd
from qlearning import ExperienceReplay

pygame.init()

pixelsPerBlock = 50
grid_size = 10
screen = pygame.display.set_mode((grid_size * pixelsPerBlock, grid_size * pixelsPerBlock))

BLACK = 0, 0, 0
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
RAY = (0, 255, 0)

displayGrid = True

game = Explo1(grid_size)


def drawGrid(grid_size, prixelsPerBlock):
    for i in range(grid_size):
        startH = np.array((0, pixelsPerBlock)) * i
        stopH = np.array((grid_size * pixelsPerBlock, pixelsPerBlock)) * i
        startV = np.array((pixelsPerBlock, 0)) * i
        stopV = np.array((pixelsPerBlock, pixelsPerBlock * pixelsPerBlock)) * i
        pygame.draw.line(screen, WHITE, startH, stopH)
        pygame.draw.line(screen, WHITE, startV, stopV)


def drawMap(map):
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j] == 1:
                start = np.array((i, j)) * pixelsPerBlock
                dim = [pixelsPerBlock, pixelsPerBlock]
                pygame.draw.rect(screen, YELLOW, [start, dim])

def drawRay(surface, ray):
    pygame.draw.polygon(surface, BLACK, ray, 0)


while 1:
    pygame.time.wait(100)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    robotPos, robotDir, state, ray_vision = game.getGameData()
    # pos = np.array((0,0))
    # pos[0] = int(robotPos[0] * pixelsPerBlock)
    # pos[1] = int(robotPos[1] * pixelsPerBlock)

    screen.fill(BLACK)
    if displayGrid:
        drawGrid(grid_size, pixelsPerBlock)
    drawMap(state)
    reward = game.play(1)
    pygame.draw.circle(screen, RED, robotPos + pixelsPerBlock/2, pixelsPerBlock/2)
    startDir = robotPos + pixelsPerBlock/2
    stopDir = startDir + robotDir*pixelsPerBlock
    pygame.draw.line(screen, RED, startDir, stopDir)
    light = pygame.Surface((grid_size * pixelsPerBlock, grid_size * pixelsPerBlock))
    light.fill(pygame.color.Color('Grey'))
    drawRay(light, ray_vision)
    screen.blit(light, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
    pygame.display.flip()

