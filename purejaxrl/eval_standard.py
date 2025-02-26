import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from luxai_s3.wrappers import LuxAIS3GymEnv
from typing import Any
import jax, chex
import jax.numpy as jnp
from purejaxrl.utils import EnvObs_to_dict
from tqdm import tqdm

def run_arena_standard_agents(
    gym_env: LuxAIS3GymEnv,
    agent_0_instantiator: Any,
    agent_1_instantiator: Any,
    use_tdqm=False
):
    assert gym_env.numpy_output
    obs, info = gym_env.reset()
    agent_0 = agent_0_instantiator(gym_env.env_params)
    agent_1 = agent_1_instantiator(gym_env.env_params)
    max_steps = (gym_env.env_params.max_steps_in_match + 1) * gym_env.env_params.match_count_per_episode

    for step in tqdm(range(max_steps), desc = f"Recording a match {agent_0.__class__.__name__} vs  {agent_1.__class__.__name__}", disable = not use_tdqm):
        action = {
            "player_0": agent_0.act(step=step, obs=obs["player_0"]),
            "player_1": agent_1.act(step=step, obs=obs["player_1"]),
        }
        obs, reward, _, _, _ = gym_env.step(action)
    return {"player_0": obs["player_0"]["team_wins"][0], "player_1": obs["player_0"]["team_wins"][1]}

def test_c():
    from rule_based.relicbound.agent import RelicboundAgent
    from rule_based.random.agent import RandomAgent
    from purejaxrl.purejaxrl_agent import PureJaxRLAgent

    agent_0_instantiator = lambda env_params: PureJaxRLAgent("player_0", env_params.__dict__)
    agent_1_instantiator = lambda env_params: RelicboundAgent("player_1", env_params.__dict__)
    gym_env = LuxAIS3GymEnv(numpy_output=True)

    reward = run_arena_standard_agents(
        agent_0_instantiator=agent_0_instantiator, agent_1_instantiator=agent_1_instantiator, gym_env=gym_env, use_tdqm=False
    )

    print(reward)

if __name__ == "__main__":
    for _ in tqdm(range(10)): test_c()