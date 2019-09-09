import numpy as np
import torch


class RolloutWorker:
    def __init__(self, env, agent, args):
        self.env = env
        self.agent = agent
        self.episode_limit = args.episode_limit
        self.obs_shape = args.obs_shape
        self.args = args

    def generate_episode(self, noise, epsilon):
        obs, obs_next, acts, r, goals, achieved_goals = [], [], [], [], [], []
        # reset the environment
        state = self.env.reset()
        # start to collect samples
        for t in range(self.episode_limit):
            # self.env.render()
            o, ag, g = state['observation'], state['achieved_goal'], state['desired_goal']
            with torch.no_grad():
                action = self.agent.select_actions(o, g, noise, epsilon)
            new_state, _, _, info = self.env.step(action)
            # print(r)
            obs.append(o.copy().reshape(-1, self.args.obs_shape))
            achieved_goals.append(ag.copy().reshape(-1, self.args.goal_shape))
            goals.append(g.copy().reshape(-1, self.args.goal_shape))
            acts.append(action.copy().reshape(-1, self.args.action_shape))

            state = new_state
        obs.append(o.copy().reshape(-1, self.args.obs_shape))
        achieved_goals.append(ag.copy().reshape(-1, self.args.goal_shape))
        # obs_1是一个长度为51的列表，列表里装着51个shape为(1, 10)的obs， acts长度为50
        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals
                       )
        # for key in episode.keys():
        #     episode[key] = np.array([episode[key]])
        return episode

    # do the evaluation
    def evaluate(self, render=False):
        success_rate = []
        for _ in range(self.args.n_cycles_evaluate):
            count = 0
            state = self.env.reset()
            for _ in range(self.args.episode_limit):
                if render:
                    self.env.render()
                o, g = state['observation'], state['desired_goal']
                with torch.no_grad():
                    action = self.agent.select_actions(o, g, 0, 0)

                new_state, _, _, info = self.env.step(action)
                state = new_state
            if info['is_success'] == 1:
                count = 1
            # print('Test finished, count = ', count)
        success_rate.append(info['is_success'])
        s = [i for i in success_rate if i == 1]
        return len(s) / len(success_rate)
