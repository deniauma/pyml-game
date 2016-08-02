import sys, pygame, random, math
import numpy as np
from game import Explo1

pygame.init()

pixelsPerBlock = 50
grid_size = 10
screen = pygame.display.set_mode((grid_size * pixelsPerBlock, grid_size * pixelsPerBlock))

BLACK = 0, 0, 0
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

displayGrid = True

game = Explo1(grid_size)


while 1:
    pygame.time.wait(10)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    robotPos, robotDir, state = game.getGameData()
    pos = robotPos * pixelsPerBlock


    screen.fill(BLACK)
    if displayGrid:
        for i in range(grid_size):
            startH = np.array((0, pixelsPerBlock)) * i
            stopH = np.array((grid_size*pixelsPerBlock, pixelsPerBlock)) * i
            startV = np.array((pixelsPerBlock, 0)) * i
            stopV = np.array((pixelsPerBlock, pixelsPerBlock * pixelsPerBlock)) * i
            pygame.draw.line(screen, WHITE, startH, stopH)
            pygame.draw.line(screen, WHITE, startV, stopV)
    pygame.draw.circle(screen, RED, pos + pixelsPerBlock/2, pixelsPerBlock/2)
    startDir = pos + pixelsPerBlock/2
    stopDir = startDir + robotDir*pixelsPerBlock
    pygame.draw.line(screen, RED, startDir, stopDir)
    # pygame.draw.rect(screen, mouseColor, [mousePos,mouseDim])
    pygame.display.flip()
