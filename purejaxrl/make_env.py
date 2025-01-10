import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from typing import Any
import jax
import numpy as np

# base env
from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env, EnvObs, PlayerName, EnvParams

# wrappers
from purejaxrl.wrappers.base_wrappers import LogWrapper, SimplifyTruncation, SimplifyAction
from purejaxrl.wrappers.record_wrapper import RecordEpisode
# obs
from purejaxrl.wrappers.obs_wrappers import TransformObsWrapper
from purejaxrl.wrappers.transform_obs import HybridTransformObs
# reward
from purejaxrl.wrappers.reward_wrappers import TransformRewardWrapper
from purejaxrl.wrappers.transform_reward import BasicPointBasedReward
from sample_params import sample_params


def make_env(record = False, **record_kwargs):
    #Wrappers
    env = LuxAIS3Env(auto_reset=False, fixed_env_params=EnvParams())
    env = SimplifyTruncation(env) # always has to be first !
    if record:
        env = RecordEpisode(env, **record_kwargs)

    env = SimplifyAction(env)
    env = TransformObsWrapper(env, HybridTransformObs())
    env = TransformRewardWrapper(env, BasicPointBasedReward())
    env = LogWrapper(env) # always has to be last ! 
    return env

if __name__ == "__main__":
    env = make_env(record = True, save_dir = "here", save_format = "html")
    key = jax.random.PRNGKey(0)
    params = sample_params(key)
    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, {player: env.action_space.sample(key) for player in env.players}, params=params)
    env.close()
    print("reward:", reward)
