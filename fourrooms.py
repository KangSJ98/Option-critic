import numpy as np
import pygame
import gym
from gym import spaces
import pygame
import time

class Fourrooms(gym.Env):
    metadata =  {"render_mode": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        # map : 기본 틀
        self.map = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        self.observation_space = spaces.Box(low=0., high=1., shape=(self.map.size,))

        # ex) tostate[(2,3)] = 14, tocell(14) = (2,3)
        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                self.tostate[(i,j)] = statenum
                statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        # 빈 공간만 시작 위치로 설정
        self.goal = self.tostate[(7, 9)]
        empty_spaces = [self.tostate[(i, j)] for (i, j) in self.tostate.keys() if self.map[i, j] == 0]
        self.init_state = empty_spaces.copy()
        self.init_state.remove(self.goal)

        self.rng = np.random.RandomState(int(time.time()))

        # ep_steps : 에피소드 길이, currentcell : agent 위치
        self.ep_steps = 0
        self.currentcell = None

        # 상하좌우
        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]

    # 초기화
    def reset(self):
        state = self.rng.choice(self.init_state)
        self.currentcell = self.tocell.get(state)

        self.ep_steps = 0
        return self.get_state(state)

    # state 입력 받아서 map에서 state 위치만 2로 바꾼 후 반환
    def get_state(self, state):
        s = self.map.copy()
        pos = self.tocell[state]
        s[pos[0], pos[1]] = 2
        return s

    # cell 입력 받아서 상화좌우 중 빈 공간 반환
    def empty_around(self, cell):
        avail = []
        for action in range(4):
            nextcell = tuple(cell + self.directions[action])
            if not self.map[nextcell]:
                avail.append(nextcell)
        return avail
    
    # goal 위치 변경
    def switch_goal(self):
        prev_goal = self.goal
        self.goal = self.rng.choice(self.init_state)
        self.init_state.append(prev_goal)
        self.init_state.remove(self.goal)

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        self.ep_steps += 1

        nextcell= tuple(self.currentcell + self.directions[action])
        # 다음 위치가 비어있으면 2/3 선택한 방향으로 이동, 1/3 랜덤하게 이동.
        if not self.map[nextcell]:
            if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
            else:
                self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = float(done)

        if not done and self.ep_steps >= 1000:
            done = True
            reward = 0.0

        return self.get_state(state), reward, done, None