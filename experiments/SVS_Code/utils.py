def from_env_name_to_class(env_name):
    import importlib
    module = importlib.import_module('gym_pybullet_drones.envs.multi_agent_rl.' + env_name)
    env_class = getattr(module, env_name)
    return env_class

def from_env_name_to_class_test(env_name):
    import importlib
    module = importlib.import_module('gym_pybullet_drones.envs.multi_agent_rl.test_simpler_env.' + env_name)
    env_class = getattr(module, env_name)
    return env_class


def build_env_by_name(env_class, exp, **kwargs):
    temp_kwargs = kwargs.copy()

    if exp:
        kwargs["gui"] = False  # This will avoid two spawned gui
    else:
        temp_kwargs["gui"] = False  # This will avoid two spawned gui
    
    temp_env = env_class(exp, **temp_kwargs)
    return lambda _: env_class(exp, **kwargs), temp_env.observation_space, temp_env.action_space, temp_env
