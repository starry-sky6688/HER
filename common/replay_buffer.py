import threading
import numpy as np


class Buffer:
    def __init__(self, args, sample_func):
        """
        sample_func: the function to convert the data to the type og her

        因为her要将经验中的goal转化成当前episode中在该经验之后达到的obs，所以需要以episode为单位来存，如果以transition
        为单位来存，就不知道它的后面到达的obs是哪个了

        """

        self.T = args.episode_limit
        self.size = args.buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.T + 1, args.obs_shape]),
                        'ag': np.empty([self.size, self.T + 1, args.goal_shape]),
                        'g': np.empty([self.size, self.T, args.goal_shape]),
                        'u': np.empty([self.size, self.T, args.action_shape])
                        }
        # thread lock
        self.lock = threading.Lock()
        self.args = args

    # store the episode
    def store_episode(self, episode_batch):
        o, ag, g, u = episode_batch['o'], episode_batch['ag'], episode_batch['g'], episode_batch['u']
        batch_size = o.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = o
            self.buffers['ag'][idxs] = ag
            self.buffers['g'][idxs] = g
            self.buffers['u'][idxs] = u
            self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        # 这里不需要随机抽取batch_size条经验，直接拿buffer里所有的经验，在sample_func()里对经验进行转换时会进行抽取
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['o_next'] = temp_buffers['o'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
