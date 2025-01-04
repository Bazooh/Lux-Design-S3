from typing import Any
import jax
import numpy as np

# base env
from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env, EnvObs, PlayerName

# wrappers
from purejaxrl.wrappers.obs_wrappers import TransformObservation
from purejaxrl.wrappers.reward_wrappers import TransformReward
from purejaxrl.wrappers.base_wrappers import LogWrapper, SimplifyTruncation

@jax.jit
def transform_obs(observation: EnvObs):
    return(jax.numpy.arange(24, dtype='float32'))

@jax.jit
def transform_reward(reward: Any):
    return(0)


def make_env(seed: int):
    
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

if __name__ == "__main__":
    env = make_env(0)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    print(obs)