import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import jax
from purejaxrl.env.utils import sample_params
from purejaxrl.parse_config import parse_config
import numpy as np

# base env and wrappers
from luxai_s3.env import LuxAIS3Env
from purejaxrl.env.wrappers import *

def make_env(env_args, auto_reset = False, record=False, **record_kwargs) -> TransformObsWrapper:
    env = LuxAIS3Env(auto_reset=auto_reset)
    env = PointsMapWrapper(env)
    if record:
        env = RecordEpisodeWrapper(env, **record_kwargs)
    env = SimplifyTruncationWrapper(env)
    env = MemoryWrapper(env, env_args["memory"])
    env = TrackerWrapper(env)
    env = TransformRewardWrapper(env, reward_phases = env_args["reward_phases"], reward_smoothing = env_args["reward_smoothing"])
    env = TransformActionWrapper(env, env_args["transform_action"])
    env = TransformObsWrapper(env, env_args["transform_obs"])
    return env


def make_vanilla_env(env_args, auto_reset = False, record=False, **record_kwargs) -> TrackerWrapper:
    env = LuxAIS3Env(auto_reset=auto_reset)
    env = PointsMapWrapper(env)
    if record:
        env = RecordEpisodeWrapper(env, **record_kwargs)
    env = SimplifyTruncationWrapper(env)
    env = MemoryWrapper(env, env_args["memory"])
    env = TrackerWrapper(env)
    env = TransformRewardWrapper(env, reward_phases = env_args["reward_phases"], reward_smoothing = env_args["reward_smoothing"])
    return env


if __name__ == "__main__":
    config = parse_config()
    seed = np.random.randint(0, 100)
    key = jax.random.PRNGKey(seed)
    env = make_env(config["env_args"])
    env = LogWrapper(env)
    params = sample_params(key)
    obs, state = env.reset(key, params)
    rng_0, rng_1 = jax.random.split(key)
    key_a_0, key_a_1 = jax.random.split(rng_0, 16), jax.random.split(rng_1, 16)

    for i in range(0, 510):
        print("Step:", i)
        a = {"player_0": jax.vmap(env.action_space().sample)(key_a_0), "player_1": jax.vmap(env.action_space().sample)(key_a_1)}         
        obs, state, reward, done, info = env.step(key, state, a, params=params)
        print(f"info: global timestep {state.steps}, done {done}, reward {reward} ")
        print(f" episode_return_player_0 {info['episode_return_player_0']}, episode_stats_player_0 {info['episode_stats_player_0']}")
        print(f" episode_return_player_1 {info['episode_return_player_1']}, episode_stats_player_1 {info['episode_stats_player_1']}")
        if done:
            obs, state = env.reset(key, params)
            print("reset")