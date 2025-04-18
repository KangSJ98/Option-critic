import numpy as np
import torch
import argparse
from copy import deepcopy

from fourrooms import Fourrooms
from option_critic import OptionCriticFeatures
from experience_replay import ReplayBuffer
from logger import Logger
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn
from utils import to_tensor

import time
import os

parser = argparse.ArgumentParser(description="Option Critic SJ")
parser.add_argument('--env', default='Fourrooms', help='ROM to run')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=4, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(4e6), help='number of maximum steps to take.') # bout 4 million
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
parser.add_argument('--switch-goal', type=bool, default=True, help='switch goal after 2k eps')


def run(args):
    env = Fourrooms()
    option_critic = OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

    option_critic = option_critic(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device
    )

    # 학습 과정 안정화를 위해 prime network 생성
    option_critic_prime = deepcopy(option_critic)

    # 최적화 대상 지정
    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    env.init_render()

    # replay buffer, logger
    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = Logger(logdir=args.logdir, run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{time.ctime()}")

    steps = 0
    if args.switch_goal:
        print("current goal : ", env.goal)

    while steps < args.max_steps_total:
        obs = env.reset()
        state = option_critic.get_state(to_tensor(obs))
        greedy_option = option_critic.greedy_option(state)
        rewards = 0; ep_steps = 0; done = False

        # option
        option_length = {opt:[] for opt in range(args.num_options)}
        greedy_option = option_critic.greedy_option(state)
        current_option = 0; option_termination = True; curr_op_len = 0

        # change goal at 2000 episod
        if args.switch_goal and logger.n_eps == 2000:
            env.switch_goal()
            print("New goal ", env.goal)
        if logger.n_eps > 4000:
            break

        # episod
        while not done and ep_steps < args.max_steps_ep:
            # 200번 마다 렌더링
            if logger.n_eps % 200 == 199:
                env.render()
            epsilon = option_critic.epsilon

            # 옵션 종료 시 새로운 옵션 선택(e-greedy)
            if option_termination:
                option_length[current_option].append(curr_op_len)
                # e-greedy 
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0
            
            # choose action according state, option
            # action_dist의 entropy와 선택된 action의 로그 확률
            action, logp, entropy = option_critic.get_action(state, current_option)

            # take action
            next_obs, reward, done, _ = env.step(action)

            # replay buffer에 (obs, 현재 option, reward, next state, done) 저장
            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            actor_loss, critic_loss = None, None
            # buffer 크기가 batch size보다 커지면
            if len(buffer) > args.batch_size:
                # actor loss
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, reward, done, next_obs, option_critic, option_critic_prime, args)
                loss = actor_loss

                # critic loss
                # 주기를 가지는 이유 : computation, stability 단점 : 최신 데이터 반영 x
                if steps % args.update_frequency == 0:
                    buffer_data = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, buffer_data, args)
                    loss += critic_loss

                # gradient 0으로 초기화
                optim.zero_grad()
                # loss(actor loss + critic loss)로 backpropagation(모든 매개변수에 대한 gradient 계산)
                loss.backward()
                # 계산된 gradient로 매개변수 업데이트
                optim.step()

                # prime network에 복사
                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))

            # update steps, etc
            steps += 1; ep_steps += 1; curr_op_len += 1; obs = next_obs

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

        logger.log_episode(steps, rewards, option_length, ep_steps, epsilon)

if __name__=="__main__":
    args = parser.parse_args()
    run(args)