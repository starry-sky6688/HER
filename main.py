import numpy as np
import gym
import os
from common.arguments import get_args
import random
import torch
from common.utils import get_env_params
from runner import Runner


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    args = get_args()
    env = gym.make(args.env_name)
    # set random seeds for reproduce
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # get the environment parameters
    env_params = get_env_params(env)
    args.obs_shape = env_params['obs_shape']
    args.goal_shape = env_params['goal_shape']
    args.action_shape = env_params['action_shape']
    args.action_max = env_params['action_max']
    args.episode_limit = env_params['episode_limit']
    runner = Runner(args, env)
    runner.run()

