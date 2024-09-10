from functools import partial
import sys
sys.path.append("./smac")
from smac.env import MultiAgentEnv, StarCraft2Env
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)