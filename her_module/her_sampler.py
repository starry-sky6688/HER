import numpy as np
'''
在DDPG里，store_episode()和sample_batch()里都用了这个函数，但是前者的batch_size是所有transition的个数，
用来把所有的transition都进行修改，后者是真的要用来经验回放的transition个数
'''


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class HerSampler:
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        #### HER经验回放是DDPG 经验回放的次数的四倍
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    '''
    在函数中定义一个函数，首先将make_sample_her_transitions当作一个函数执行，
    返回的是_sample_her_transitions函数对象，因为没有执行他
    然后调用make_sample_her_transitions时只需要传入_sample_her_transitions的参数即可
    '''

    # 返回的transition为一个dict，字典中obs的shape为(batch_size_in_transitions, obs.dim)
    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        """
        episode_batch是以{'o', 'u', 'g', 'ag'}的形式存的
        每个key：value的形式为  key: array(T x dim_key)
        """

        T = episode_batch['o_next'].shape[1]  # 得到buffer里每个episode的长度
        rollout_batch_size = episode_batch['u'].shape[0]  # 得到buffer里episode的个数
        # batch_size是要比T大的，他是整个episode_batch中transition的个数，从而对一条经验修改多次
        batch_size = batch_size_in_transitions

        # 从episode_batch中取出batch_size条经验
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)  # 取batch_size个第一维坐标
        t_samples = np.random.randint(T, size=batch_size)  # 取batch_size个第二维坐标,即要取哪些经验，然后用future_p决定是否修改
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}  # 取出想要的transitions，shape为(49,10)，变成二维的49条经验

        # episode_idxs[her_indexes]表示要修改经验的episode
        # t_sample是所有抽取的batch_size条经验在各自episode中的位置
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)  # 用概率future_p决定用哪些经验进行修改
        # future_offset是相对于t_samples的偏移，因为修改时要把goal改成后面的某个achieved_goal
        # T是episode的长度，T - t_samples表示每条经验后面还有多少条
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        # (t_samples + 1 + future_offset)是每个t_samples对应的后面的某条经验，再取[her_indexes]就得到了对应的要改变的位置
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # 得到第episode_idxs[her_indexes]条episode中第future_t个transition上的achieved_goal
        # 此时future_ag的数量是her_indexes的长度，它是比transition数量少五分之一的
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]

        # 把goal改成了future_t上的achieved_goal
        for index, her_index in enumerate(her_indexes[0]):  # her_indexes的shape是tuple:(array(),)，因此要取tuple[0]
            transitions['g'][her_index] = future_ag[index]  # 把第t_samples个transition上的achieved_goal换乘第future_t个的

        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        transitions['r'] = self.reward_func(transitions['ag_next'], transitions['g'], info)
        # transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
        #                for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions
