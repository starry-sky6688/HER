import numpy as np
import inspect
import functools


# 在HER中没有转换之前的episode字典中的obs，是一个[50*10]的列表，转化之后是一个[1,50,10]的向量
def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    return episode_batch


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = dict(obs_shape=obs['observation'].shape[0],
                  goal_shape=obs['desired_goal'].shape[0],
                  action_shape=env.action_space.shape[0],
                  action_max=env.action_space.high[0],
                  episode_limit=env._max_episode_steps
                  )
    return params


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def clip_og(o, g, clip_obs):
    o = np.clip(o, -clip_obs, clip_obs)
    g = np.clip(g, -clip_obs, clip_obs)
    return o, g
