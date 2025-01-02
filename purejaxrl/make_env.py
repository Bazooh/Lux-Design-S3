from typing import Any
import jax
import numpy as np

# base env
from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env, EnvObs, PlayerName

# wrappers
from obs_wrappers import TransformObservation
from reward_wrappers import TransformReward
from base_wrappers import LogWrapper, SimplifyTruncation

@jax.jit
def transform_obs(observation: dict[PlayerName, EnvObs]):
    return(jax.numpy.zeros(24, dtype='float32'))

@jax.jit
def transform_reward(reward: Any):
    return(0)


def make_env(seed: int, num_envs: int):
    
    seed = 0
    np.random.seed(seed)
    rng_key = jax.random.key(seed)
    
    # sample random params initially
    rng_key, subkey = jax.random.split(rng_key)
    
    #Wrappers
    env = LuxAIS3Env(auto_reset=False, fixed_env_params=EnvParams())
    env = SimplifyTruncation(env) # always has to be first !
    env = TransformObservation(env, transform_obs)
    env = TransformReward(env, transform_reward)
    env = LogWrapper(env) # always has to be last ! 
    return env