from agents.base_agent import Agent, N_Actions, N_Agents
from agents.obs import EnvParams, Obs
from typing import Any
import jax
from functools import partial
from utils import sample_action
import chex

def symetric_action_not_vectorized(action: int):
    return [0, 3, 4, 1, 2][action]

class JaxAgent(Agent):
    def __init__(
        self,
        player: str,
        env_params: EnvParams,
        network: Any,
        network_params: Any,
        env: Any,
        key : chex.PRNGKey,
    ):
        super().__init__(player, env_params)
        self.network = network
        self.network_params = network_params
        self.key = key

    @partial(jax.jit, static_argnums=(0, 1))
    def sample_actions(
        self, 
        key: chex.PRNGKey,
        obs: Any,
    ):
        logits, value = self.network.apply(self.network, **obs) # probs is (16, 5)
        action = sample_action(key, logits)
        return action

    def actions(
        self, 
        obs,
    ):
        return self.sample_actions(self.key, obs)

    def act(
        self, 
        step: int,
        obs: dict[str, Any], 
        remainingOverageTime: int = 60,
    ):
        return self.actions(Obs.from_dict(obs), remainingOverageTime)
