import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pygame
import time
import cv2
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'

logger = logging.getLogger(__name__)

class Fourrooms(gym.Env,):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

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
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])

        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0., high=1., shape=(np.sum(self.occupancy == 0),))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}

        self.goal = 62 # East doorway
        self.init_states = list(range(self.observation_space.shape[0]))
        self.init_states.remove(self.goal)
        self.ep_steps = 0
        self.currentcell = None

        self.reset()
        self.init_render()

    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell.get(state)
        self.ep_steps = 0
        return self.get_state(state)

    def switch_goal(self):
        prev_goal = self.goal
        self.goal = self.rng.choice(self.init_states)
        self.init_states.append(prev_goal)
        self.init_states.remove(self.goal)
        assert prev_goal in self.init_states
        assert self.goal not in self.init_states

    def get_state(self, state):
        s = np.zeros(self.observation_space.shape[0])
        s[state] = 1
        return s


    def init_render(self):
        # Pygame 초기화
        pygame.init()

        # 윈도우 크기 설정
        self.window_size = (400, 400)
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Fourrooms Environment")

        # 타일 크기 계산
        self.tile_size = (self.window_size[0] // self.occupancy.shape[1], self.window_size[1] // self.occupancy.shape[0])

    def render(self, show_goal=True):
        # 새로운 화면
        screen = pygame.Surface(self.window_size)


        # 그리드
        for i in range(self.occupancy.shape[0]):
            for j in range(self.occupancy.shape[1]):
                rect = pygame.Rect(j * self.tile_size[0], i * self.tile_size[1], self.tile_size[0], self.tile_size[1])
                if self.occupancy[i, j] == 1:
                    # 벽
                    pygame.draw.rect(screen, (0, 0, 0), rect)
                else:
                    # 빈 공간
                    pygame.draw.rect(screen, (255, 255, 255), rect)

        # 에이전트 위치 그리기
        agent_cell = self.currentcell
        if agent_cell is not None:
            rect = pygame.Rect(
                agent_cell[1] * self.tile_size[0],
                agent_cell[0] * self.tile_size[1],
                self.tile_size[0],
                self.tile_size[1]
            )
            pygame.draw.rect(screen, (255, 0, 0), rect)
            


        if show_goal:
            # 목표 위치 그리기
            goal_cell = self.tocell[self.goal]
            rect = pygame.Rect(goal_cell[1] * self.tile_size[0], goal_cell[0] * self.tile_size[1],
                            self.tile_size[0], self.tile_size[1])
            pygame.draw.rect(screen, (0, 255, 0), rect)

        # 실제 화면에 업데이트
        self.window.blit(screen, (0, 0))
        pygame.display.flip()
        time.sleep(0.05)

        return self.window


    def record_step(self, screen):
        screen_array = pygame.surfarray.array3d(screen)
        screen_bgr = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)

        screen_bgr = cv2.transpose(screen_bgr)

        if self.video_out is None:
            fps = 30
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_out = cv2.VideoWriter(self.video_filename, fourcc, fps, self.window_size)

        self.video_out.write(screen_bgr)

    def record_start(self, filename):
        self.video_filename = filename
        self.video_out = None

    def record_end(self):
        if self.video_out is not None:
            self.video_out.release()
            self.video_out = None


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

        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            if self.rng.uniform() < 1/3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
            else:
                self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = state == self.goal
        reward = float(done)

        if not done and self.ep_steps >= 1000:
            done = True ; reward = 0.0

        

        return self.get_state(state), reward, done, None

if __name__=="__main__":
    env = Fourrooms()
    env.seed(3)
