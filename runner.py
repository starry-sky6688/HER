from tqdm import tqdm
from common.replay_buffer import Buffer
from common.rollout import RolloutWorker
from agent import Agent
from her_module.her_sampler import HerSampler
from common.utils import convert_episode_to_batch_major
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env):
        self.noise = args.noise_eps
        self.epsilon = args.epsilon
        self.env = env
        self.agent = Agent(args)
        self.her_module = HerSampler(args.replay_strategy, args.replay_k, env.compute_reward)
        self.buffer = Buffer(args, self.her_module.sample_her_transitions)
        self.worker = RolloutWorker(self.env, self.agent, args)
        self.args = args

    def run(self):
        success_rates = []
        for epoch in tqdm(range(self.args.n_epochs)):
            for episode_idx in range(self.args.n_cycles):
                episode = self.worker.generate_episode(self.noise, self.epsilon)
                episode_batch = convert_episode_to_batch_major(episode)  # 把episode中的二维数据变成三维的
                self.buffer.store_episode(episode_batch)
                episode_batch['o_next'], episode_batch['ag_next'] = episode_batch['o'][:, 1:], episode_batch['ag'][:, 1:]
                transitions = self.her_module.sample_her_transitions(episode_batch, self.args.episode_limit)

                # update the normalizer
                self.agent.update_normalizer(transitions)

            for _ in range(self.args.n_batches):
                transitions = self.buffer.sample(self.args.batch_size)
                self.agent.learn(transitions)
            # self.noise = max(0, self.noise - 0.001)
            # self.epsilon = max(0.05, self.noise - 0.001)
            if len(success_rates) > 0 and success_rates[-1] > 0.5:
                success_rate = self.worker.evaluate(render=True)
            else:
                success_rate = self.worker.evaluate()
            success_rates.append(success_rate)
        save_path = self.args.save_dir + '/' + self.args.env_name
        plt.figure()
        plt.plot(range(self.args.n_epochs), success_rates)
        plt.xlabel('epoch')
        plt.ylabel('success_rate')
        plt.savefig(save_path + '/plt.png', format='png')

