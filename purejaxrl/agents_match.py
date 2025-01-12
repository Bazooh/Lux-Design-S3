import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax, chex
from typing import Any
import jax.numpy as jnp
from utils import sample_params
from luxai_s3.env import LuxAIS3Env, EnvObs
import numpy as np
import termplotlib as tpl
import numpy as np
import flax
from luxai_s3.utils import to_numpy
def EnvObs_to_dict(obs: EnvObs) ->  dict[str, Any]:
    return {
        "units": {
            "position": obs.units.position,
            "energy": obs.units.energy,
        },
        "units_mask": obs.units_mask,
        "sensor_mask": obs.sensor_mask,
        "map_features": {
            "energy": obs.map_features.energy,
            "tile_type": obs.map_features.tile_type,
        },
        "relic_nodes": obs.relic_nodes,
        "relic_nodes_mask": obs.relic_nodes_mask,
        "team_points": obs.team_points,
        "team_wins": obs.team_wins,
        "steps": obs.steps,
        "match_steps": obs.match_steps
    }

def run_match(
        agent_0: Any, 
        agent_1: Any, 
        vanilla_env: LuxAIS3Env, 
        env_params,
        key: chex.PRNGKey,
    ):
    """
    Evaluate the trained agent against a reference agent using a separate eval environment.
    """
    rng, _rng = jax.random.split(key)
    obs, env_state = vanilla_env.reset(rng, env_params)

    max_episode_steps = (
        vanilla_env.fixed_env_params.max_steps_in_match + 1
    ) * vanilla_env.fixed_env_params.match_count_per_episode # 101 * 5 steps per env

    points = np.zeros((max_episode_steps, 2))

    for step_idx in range(max_episode_steps):
        
        action = {
            "player_0": agent_0.act(step = step_idx, obs = EnvObs_to_dict(obs["player_0"])), 
            "player_1": agent_1.act(step = step_idx, obs = EnvObs_to_dict(obs["player_1"])),
        }
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, truncated_dict, terminated_dict, info = vanilla_env.step(rng, env_state, action, env_params)
        points[step_idx] = obs["player_0"].team_points

    fig = tpl.figure()
    fig.plot(np.arange(max_episode_steps), points[:, 0], width=100, height = 15)
    fig.plot(np.arange(max_episode_steps), points[:, 1], width=100, height = 15)
    fig.show()
    return points # shape (max_episode_steps, 2)


if __name__ == "__main__":
    from purejaxrl.jax_agent import JaxAgent, RawJaxAgent
    from rule_based.random.agent import RandomAgent
    from rule_based.relicbound.agent import RelicboundAgent
    from rule_based.naive.agent import NaiveAgent
    # RUN MATCH
    seed = 1
    key = jax.random.PRNGKey(seed)
    vanilla_env = LuxAIS3Env(auto_reset=True)
    env_params = sample_params(key)
    run_match(
        agent_0 = JaxAgent("player_0", env_params.__dict__),
        agent_1 = JaxAgent("player_1", env_params.__dict__),
        key = key,
        vanilla_env = vanilla_env,
        env_params = env_params
    )