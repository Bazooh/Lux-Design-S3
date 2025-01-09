import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from typing import Any
import jax
import numpy as np

# base env
from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env, EnvObs, PlayerName, EnvParams

# wrappers
from purejaxrl.wrappers.base_wrappers import LogWrapper, SimplifyTruncation
# obs
from purejaxrl.wrappers.obs_wrappers import TransformObsWrapper
from purejaxrl.wrappers.transform_obs import HybridTransformObs
# reward
from purejaxrl.wrappers.reward_wrappers import TransformRewardWrapper
from purejaxrl.wrappers.transform_reward import BasicPointBasedReward
from sample_params import sample_params


def make_env(seed: int):
    
    seed = 0
    np.random.seed(seed)
    rng_key = jax.random.key(seed)
    
    # sample random params initially
    rng_key, subkey = jax.random.split(rng_key)
    
    #Wrappers
    env = LuxAIS3Env(auto_reset=False, fixed_env_params=EnvParams())
    env = SimplifyTruncation(env) # always has to be first !
    env = TransformObsWrapper(env, HybridTransformObs())
    env = TransformRewardWrapper(env, BasicPointBasedReward())
    env = LogWrapper(env) # always has to be last ! 
    return env

if __name__ == "__main__":
    env = make_env(0)
    key = jax.random.PRNGKey(0)
    params = sample_params(key)
    obs, state = env.reset(key, params)
    print(obs)
    print(env.observation_space.sample(key))