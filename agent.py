import numpy as np
import torch
import os
from her_module.ddpg import DDPG
from common.utils import clip_og
from common.normalizer import Normalizer


class Agent:
    def __init__(self, args):
        self.args = args
        self.policy = DDPG(args)
        # create the normalizer
        self.o_norm = Normalizer(size=args.obs_shape, default_clip_range=self.args.clip_range)
        self.g_norm = Normalizer(size=args.goal_shape, default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    # pre_process the inputs
    def preprocess_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def select_actions(self, o, g, noise_rate, random_rate):
        input_tensor = self.preprocess_inputs(o, g)
        pi = self.policy.actor_network(input_tensor)
        # print('{} : {}'.format(self.name, pi))
        u = pi.cpu().numpy()
        noise = noise_rate * self.args.action_max * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.args.action_max, self.args.action_max)
        u += np.random.binomial(1, random_rate, u.shape[0]).reshape(-1, 1) * (
                np.random.uniform(low=-self.args.action_max, high=self.args.action_max,
                                  size=(u.shape[0], 4)) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        return u.copy()

    def update_normalizer(self, transitions):
        o, g = transitions['o'], transitions['g']
        # pre process the obs and g
        transitions['o'], transitions['g'] = clip_og(o, g, self.args.clip_range)
        # update
        self.o_norm.update(transitions['o'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def learn(self, transitions):
        # clip
        o, g, o_next = transitions['o'], transitions['g'], transitions['o_next']
        transitions['o'], transitions['g'] = clip_og(o, g, self.args.clip_range)
        transitions['o_next'], _ = clip_og(o_next, g, self.args.clip_range)

        # normalize
        o, g, o_next = transitions['o'], transitions['g'], transitions['o_next']
        transitions['o'] = self.o_norm.normalize(o)
        transitions['o_next'] = self.o_norm.normalize(o_next)
        transitions['g'] = self.g_norm.normalize(g)

        self.policy.train(transitions)

