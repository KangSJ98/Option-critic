import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import pygame
import time
from logger import Logger

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
        self.action_space = spaces.Discrete(4)
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
        s = np.where(self.map.flatten() == 1, 1, 0)
        s[state] = 2
        return s
    
    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
    
    def init_render(self):
        # Pygame 초기화
        pygame.init()

        # 윈도우 크기 설정
        self.window_size = (400, 400)
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Fourrooms Environment")

        # 타일 크기 계산
        self.tile_size = (self.window_size[0] // self.map.shape[1], self.window_size[1] // self.map.shape[0])

    def render(self):
        # 새로운 화면
        screen = pygame.Surface(self.window_size)

        # grid
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                rect = pygame.Rect(j * self.tile_size[0], i * self.tile_size[1], self.tile_size[0], self.tile_size[1])
                if self.map[i, j] == 1:
                    # 벽
                    pygame.draw.rect(screen, (0, 0, 0), rect)
                else:
                    # 빈 공간
                    pygame.draw.rect(screen, (255, 255, 255), rect)

        # agent
        agent_cell = self.currentcell
        rect = pygame.Rect(agent_cell[1] * self.tile_size[0], agent_cell[0] * self.tile_size[1], self.tile_size[0], self.tile_size[1])
        pygame.draw.rect(screen, (255, 0, 0), rect)
            
        # goal
        goal_cell = self.tocell[self.goal]
        rect = pygame.Rect(goal_cell[1] * self.tile_size[0], goal_cell[0] * self.tile_size[1], self.tile_size[0], self.tile_size[1])
        pygame.draw.rect(screen, (0, 255, 0), rect)

        # 실제 화면에 업데이트
        self.window.blit(screen, (0, 0))
        pygame.display.flip()
        time.sleep(0.05)

        return self.window