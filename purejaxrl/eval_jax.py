import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gym import Env
import jax, chex
from typing import Any
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np

from purejaxrl.utils import plot_stats
from purejaxrl.env.utils import sample_params
from purejaxrl.parse_config import parse_config
from purejaxrl.env.make_env import (
    make_env,
    make_vanilla_env,
    TrackerWrapper,
    LogWrapper,
    EnvParams,
    EnvObs,
)
from purejaxrl.purejaxrl_agent import PureJaxRLAgent, RawPureJaxRLAgent, JaxAgent
from rule_based_jax.random.agent import RandomAgent_Jax
from rule_based_jax.naive.agent import NaiveAgent_Jax

from jax_tqdm import scan_tqdm


def run_arena_jax_agents(
    agent_0: JaxAgent,
    agent_1: JaxAgent,
    vanilla_env: TrackerWrapper,
    key: chex.PRNGKey,
    number_of_games=8,
    match_count_per_episode=5,
    use_tdqm=False,
):
    # INITIALIZE ENVIRONMENT
    rng, _rng = jax.random.split(key)
    rng_params = jax.random.split(rng, number_of_games)
    env_params_v = jax.vmap(
        lambda key: sample_params(key, match_count_per_episode=match_count_per_episode)
    )(rng_params)
    reset_rng = jax.random.split(rng, number_of_games)
    obs_v, env_state_v = jax.vmap(vanilla_env.reset)(reset_rng, env_params_v)
    max_steps = (
        env_params_v.max_steps_in_match[0] + 1
    ) * env_params_v.match_count_per_episode[0]

    @jax.jit
    def get_actions(
        rng,
        obs_player_0: EnvObs,
        obs_player_1: EnvObs,
        memory_state_player_0: Any,
        memory_state_player_1: Any,
        env_params: EnvParams,
    ):
        rng_0, rng_1 = jax.random.split(rng)
        action_0 = agent_0.forward(
            team_id=0,
            key=rng_0,
            obs=obs_player_0,
            memory_state=memory_state_player_0,
            env_params=env_params,
        )
        action_1 = agent_1.forward(
            team_id=1,
            key=rng_1,
            obs=obs_player_1,
            memory_state=memory_state_player_1,
            env_params=env_params,
        )
        return {"player_0": action_0, "player_1": action_1}

    @scan_tqdm(
        max_steps,
        print_rate=1,
        desc=f"Running Arena Matches {agent_0.__class__.__name__} vs  {agent_1.__class__.__name__}",
        disable=not use_tdqm,
    )
    def _env_step(runner_state, _):
        env_state_v, last_obs_v, rng, env_params_v = runner_state
        rng, _ = jax.random.split(rng)
        action_rng = jax.random.split(rng, number_of_games)
        memory_state_player_0_v = env_state_v.memory_state_player_0
        memory_state_player_1_v = env_state_v.memory_state_player_1
        action_v = jax.vmap(get_actions)(
            action_rng,
            last_obs_v["player_0"],
            last_obs_v["player_1"],
            memory_state_player_0_v,
            memory_state_player_1_v,
            env_params_v,
        )

        step_rng = jax.random.split(rng, number_of_games)
        obs_v, env_state_v, _, _, info_v = jax.vmap(vanilla_env.step)(
            step_rng, env_state_v, action_v, env_params_v
        )

        runner_state = (env_state_v, obs_v, rng, env_params_v)
        return runner_state, info_v

    runner_state, info_v = jax.lax.scan(
        _env_step,
        (env_state_v, obs_v, rng, env_params_v),
        jnp.arange(max_steps),
    )

    return info_v


def run_episode_and_record(
    rec_env: TrackerWrapper,
    agent_0: JaxAgent,
    agent_1: JaxAgent,
    key: chex.PRNGKey,
    match_count_per_episode=5,
    plot_stats_curves: bool = False,
    return_states: bool = False,
    use_tdqm=False,
):
    """
    Evaluate the trained agent against a reference agent using a separate eval environment.
    """
    rng, _rng = jax.random.split(key)
    rng_params, _rng = jax.random.split(rng)
    env_params = sample_params(
        rng_params, match_count_per_episode=match_count_per_episode
    )
    reset_rng, _rng = jax.random.split(rng)
    obs, env_state = rec_env.reset(reset_rng, env_params)
    max_steps = (env_params.max_steps_in_match + 1) * env_params.match_count_per_episode

    @jax.jit
    def get_actions(
        rng,
        obs_player_0: EnvObs,
        obs_player_1: EnvObs,
        memory_state_player_0: Any,
        memory_state_player_1: Any,
        env_params: EnvParams,
    ):
        rng_0, rng_1 = jax.random.split(rng)
        action_0 = agent_0.forward(
            team_id=0,
            key=rng_0,
            obs=obs_player_0,
            memory_state=memory_state_player_0,
            env_params=env_params,
        )
        action_1 = agent_1.forward(
            team_id=1,
            key=rng_1,
            obs=obs_player_1,
            memory_state=memory_state_player_1,
            env_params=env_params,
        )
        return {"player_0": action_0, "player_1": action_1}

    stack_stats = []
    stack_states = []
    stack_vec = []
    points_map = []
    
    for _ in tqdm(range(max_steps), desc = f"Recording a match {agent_0.__class__.__name__} vs  {agent_1.__class__.__name__}", disable = not use_tdqm):
        rng, _ = jax.random.split(rng)
        
        memory_state_player_0 = env_state.memory_state_player_0
        memory_state_player_1 = env_state.memory_state_player_1
        action = get_actions(
            rng,
            obs["player_0"],
            obs["player_1"],
            memory_state_player_0,
            memory_state_player_1,
            env_params,
        )

        rng_step, _rng = jax.random.split(rng)
        obs, env_state, _, _, info = rec_env.step(
            rng_step, env_state, action, env_params
        )
        
        if return_states:
            transformed_obs_0 = agent_0.transform_obs.convert(
                team_id=0,
                obs=obs["player_0"],
                memory_state=env_state.memory_state_player_0,
                params=env_params,
            )
            transformed_obs_1 = agent_0.transform_obs.convert(
                team_id = 1, 
                obs = obs["player_1"], 
                memory_state = env_state.memory_state_player_1,
                params = env_params
            )
            stack_states.append((transformed_obs_0["image"], transformed_obs_1["image"]))
            stack_vec.append((transformed_obs_0["vector"], transformed_obs_1["vector"]))
            stack_stats.append((info["episode_stats_player_0"], info["episode_stats_player_1"]))
            points_map.append(env_state.points_map)

    rec_env.close()

    if return_states:
        stats_arrays = {
            "episode_stats_player_0": {
                stat: np.array(
                    [getattr(stack_stats[i][0], stat) for i in range(len(stack_stats))]
                )
                for stat in rec_env.stats_names
            },
            "episode_stats_player_1": {
                stat: np.array(
                    [getattr(stack_stats[i][1], stat) for i in range(len(stack_stats))]
                )
                for stat in rec_env.stats_names
            },
        }
        vec_arrays = {
            "obs_player_0": {
                feat: np.array(
                    [stack_vec[i][0][feat_idx] for i in range(len(stack_vec))],
                    dtype=np.float32,
                )
                for feat_idx, feat in enumerate(agent_0.transform_obs.vector_features)
            },
            "obs_player_1": {
                feat: np.array(
                    [stack_vec[i][1][feat_idx] for i in range(len(stack_vec))],
                    dtype=np.float32,
                )
                for feat_idx, feat in enumerate(agent_0.transform_obs.vector_features)
            },
        }

        channels_arrays = {
            "obs_player_0": {
                feat: np.array(
                    [stack_states[i][0][feat_idx] for i in range(len(stack_states))],
                    dtype=np.float32,
                )
                for feat_idx, feat in enumerate(agent_0.transform_obs.image_features)
            },
            "obs_player_1": {
                feat: np.array(
                    [stack_states[i][1][feat_idx] for i in range(len(stack_states))],
                    dtype=np.float32,
                )
                for feat_idx, feat in enumerate(agent_0.transform_obs.image_features)
            },
        }
        if plot_stats_curves:
            plot_stats(stats_arrays)

        return channels_arrays, vec_arrays, stats_arrays, points_map
    
def test_a():
    config = parse_config()
    seed = np.random.randint(0, 100)
    key = jax.random.PRNGKey(seed)
    rec_env = make_vanilla_env(
        config["env_args"],
        record=True,
        save_on_close=True,
        save_dir="test",
        save_format="html",
    )
    rec_env = LogWrapper(rec_env, replace_info=True)

    agent_0 = PureJaxRLAgent("player_0")
    agent_1 = NaiveAgent_Jax("player_1")

    run_episode_and_record(
        rec_env=rec_env, agent_0=agent_0, agent_1=agent_1, key=key, use_tdqm=True
    )


def test_b():
    config = parse_config()
    seed = np.random.randint(0, 100)
    key = jax.random.PRNGKey(seed)
    arena_env = make_vanilla_env(config["env_args"])
    arena_env = LogWrapper(arena_env)

    agent_0 = PureJaxRLAgent("player_0")
    agent_1 = config["ppo"]["arena_agent"]

    run_arena_jax_agents(
        agent_0=agent_0, agent_1=agent_1, key=key, vanilla_env=arena_env, use_tdqm=True
    )

if __name__ == "__main__":
    test_a()
    # test_b()