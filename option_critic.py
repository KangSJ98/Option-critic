import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from math import exp
import numpy as np

from utils import to_tensor

class OptionCriticFeatures(nn.Module):
    def __init__(self,
                in_features,        # 입력 개수
                num_actions,        # 액션 개수
                num_options,        # 옵션 개수
                temperature=1.0,    # softmax 액션 선택에 사용
                eps_start=1.0,      # 입실론 초기 값
                eps_min=0.1,        # 입실론 최솟값
                eps_decay=int(1e6), # 입실론 감소율
                eps_test=0.05,      # 테스트 중 사용하는 입실론 값
                device='cpu',
                testing=False):
        
        # nn 쓰기위한 초기화
        super(OptionCriticFeatures, self).__init__()

        # parameter
        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0

        # 선형 레이어
        self.features = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # option value function
        self.Q = nn.Linear(64,num_options)

        # option termination
        self.terminations = nn.Linear(64, num_options)

        # intra option policy
        # w : 64개의 input에서 num_action output으로 연결을 num_options 만큼
        # b : 각 option의 action 편향
        self.option_W = nn.Parameter(torch.zeros(num_options, 64, num_actions))
        self.option_b = nn.parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    
    # 선형 레이어 통과시킨 후 반환
    def get_state(self, obs):
        # 일반적으로 딥 러닝 모델은 배치 단위로 데이터를 처리하는데 이게 보통 4차원이라 차원을 늘려줌
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)

        return state
    
    # Q 네트워크를 통과시킨 후 반환, 현재 상태에 대한 option value function들을 계산
    def get_Q(self, state):
        return self.Q(state)
    
    # option의 종료 여부 판단, 다음 옵션 반환
    def predict_option_termination(self, state, current_option):
        # 현재 상태로 현재 선택된 옵션의 종료 확률을 출력
        # sigmoid는 종료 확률 값을 0과 1사이 값으로 변환
        termination = self.terminations(state)[:, current_option].sigmoid()
        # 0또는 1 이진으로 변환
        option_termination = Bernoulli(termination).sample()

        # 가장 큰 값의 옵션의 인덱스
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)

        return bool(option_termination.item()), next_option.item()
    
    # 각 옵션의 종료 확률 반환
    def get_terminations(self, state):
        return self.terminations(state).sigmoid()
    
    # action 선택
    def get_action(self, state, option):
        # 텐서 행렬곱, softmax 함수 사용해서 확률 분포료 변환(각 원소가 [0,1]이고 합이 1)
        # 확률 분포 기반으로 인스턴스 생성 후 샘플링
        logits = state @ self.option_W[option] + self.option_b[option]
        action_dist = logits.softmax(dim=-1)
        action_dist = Categorical(action_dist)
        action = action_dist.sample()

        # action_dist의 entropy와 선택된 action의 log확률
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action.item(), logp, entropy

    # greedy한 option 선택(Policy over option)
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()
    
    @property
    def epsilon(self):
        eps = self.eps_min + (self.eps_start- self.eps_min) * exp(-self.num_steps / self.eps_decay)
        self.num_steps += 1
        return eps
    
def critic_loss():
    