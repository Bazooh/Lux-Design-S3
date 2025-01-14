import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax
import numpy as np
from purejaxrl.utils import sample_params
import yaml

# base env and wrappers
from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env, EnvObs, PlayerName, EnvParams
from purejaxrl.wrappers.base_wrappers import LogWrapper, SimplifyTruncation, \
    TransformActionWrapper, TransformObsWrapper, TransformRewardWrapper, MemoryWrapper, SymmetryWrapper
from purejaxrl.parse_config import parse_config

def make_env(config_path = "purejaxrl/jax_config.yaml"):
    config = parse_config(config_path)
    env = LuxAIS3Env(auto_reset=True)
    env = SimplifyTruncation(env)
    env = MemoryWrapper(env, config["env_args"]["memory"])
    env = TransformRewardWrapper(env, config["env_args"]["transform_reward"])
    env = TransformActionWrapper(env, config["env_args"]["transform_action"])
    env = TransformObsWrapper(env, config["env_args"]["transform_obs"])
    #env = SymmetryWrapper(env, config["env_args"]["symmetry"])
    return env

if __name__ == "__main__":
    env = make_env()
    env = LogWrapper(env)
    key = jax.random.PRNGKey(0)
    params = sample_params(key)
    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, {player: env.action_space.sample(key) for player in env.players}, params=params)

    for i in range(1, 510):
        print("Step:", i)
        print(f"info: global timestep {info['global_timestep']}, episode timestep {info['episode_timestep']}, episode return {info['episode_return']}, episode points {info['episode_points']}, episode wins {info['episode_wins']}")
        print(state.env_state.memory_state_player_0.points_gained)
        obs, state, reward, done, info = env.step(key, state, {player: env.action_space.sample(key) for player in env.players}, params=params)
