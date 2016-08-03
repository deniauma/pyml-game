import numpy as np
from game import Game


class Explo1(Game):
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.won = False
        self._actions = {}
        self._actions[0] = "GO_STRAIGHT"
        self._actions[1] = "TURN_LEFT"
        self._actions[2] = "TURN_RIGHT"
        self._robotPos = np.array((5, 5))
        self._robotDir = np.array((0, -1))
        self._robotSpeed = 1
        self._visionDistance = 3
        self.reset()

    def reset(self):
        self._state = np.zeros((self.grid_size, self.grid_size))
        self._state[2][4] = 1
        self._state[2][5] = 1
        self._state[2][6] = 1
        self._state[6][2] = 1
        self._state[6][3] = 1
        self._state[9][7] = 1
        self._state[9][8] = 1


    @property
    def name(self):
        return "IA explo lvl 1"

    @property
    def nb_actions(self):
        return 3

    def play(self, action):
        self.moveRobot(action)

    @property
    def get_state(self):
        return self._state

    def getGameData(self):
        return self._robotPos, self._robotDir, self._state

    def rotateRobot(self, angle):
        theta = (angle / 180.) * np.pi
        x, y = self._robotDir
        xprime = x * np.cos(theta) - y * np.sin(theta)
        yprime = x * np.sin(theta) + y * np.cos(theta)
        self._robotDir = np.array((xprime, yprime))

    def checkNoCollision(self, move):
        newPos = self._robotPos + move
        if self._state[newPos[0]][newPos[1]] == 1:
            return False
        return True

    def moveRobot(self, action):
        move = np.array((0,0))
        if self._actions[action] == "GO_STRAIGHT":
            move = self._robotDir * self._robotSpeed
        elif self._actions[action] == "TURN_LEFT":
            self.rotateRobot(10)
        elif self._actions[action] == "TURN_RIGHT":
            self.rotateRobot(-10)

        if self.checkNoCollision(move):
            self._robotPos += move