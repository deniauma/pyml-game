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
FLOOR = (100,100,100)
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
    rrect = pygame.draw.polygon(surface, BLACK, ray, 0)
    # pygame.draw.rect(screen, (0,0,255), rrect)
    return rrect

def rgb2gray(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def prepross_frame(frame_rect):
    if frame_rect[0] + frame_rect[2] > grid_size * pixelsPerBlock:
        frame_rect[2] = grid_size * pixelsPerBlock - frame_rect[0]
    if frame_rect[1] + frame_rect[3] > grid_size * pixelsPerBlock:
        frame_rect[3] = grid_size * pixelsPerBlock - frame_rect[1]
    robot_vision = screen.subsurface(frame_rect)
    extract_ray = pygame.Surface((3 * pixelsPerBlock, 3 * pixelsPerBlock))
    extract_ray.fill(BLACK)
    extract_ray.blit(robot_vision, (0, 0))
    # screen.blit(extract_ray, (0, 0))
    robot_vision_pixels = pygame.surfarray.array3d(extract_ray)
    # frame_greyscale = np.zeros((3 * pixelsPerBlock, 3 * pixelsPerBlock))
    # for i in range(3 * pixelsPerBlock):
    #     for j in range(3 * pixelsPerBlock):
    #         frame_greyscale[i][j] = rgb2gray(robot_vision_pixels[i][j])


while 1:
    pygame.time.wait(100)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_UP: game.play(0)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT: game.play(1)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT: game.play(2)

    robotPos, robotDir, state, ray_vision = game.getGameData()
    # pos = np.array((0,0))
    # pos[0] = int(robotPos[0] * pixelsPerBlock)
    # pos[1] = int(robotPos[1] * pixelsPerBlock)

    screen.fill(FLOOR)
    if displayGrid:
        drawGrid(grid_size, pixelsPerBlock)
    drawMap(state)

    reward = game.play(1)    #random.randrange(3)

    pygame.draw.circle(screen, RED, robotPos + pixelsPerBlock/2, pixelsPerBlock/2)
    startDir = robotPos + pixelsPerBlock/2
    stopDir = startDir + robotDir*pixelsPerBlock
    pygame.draw.line(screen, RED, startDir, stopDir)
    light = pygame.Surface((grid_size * pixelsPerBlock, grid_size * pixelsPerBlock))
    light.fill(pygame.color.Color('Grey'))
    ray_rect = drawRay(light, ray_vision)

    prepross_frame(ray_rect)

    screen.blit(light, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

    pygame.display.flip()

