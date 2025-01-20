import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax, chex
from typing import Any
import jax.numpy as jnp
from utils import sample_action, get_logprob, get_entropy, get_obs_batch, sample_params, init_network_params, plot_stats, EnvObs_to_dict
from parse_config import parse_config
from purejaxrl.env.make_env import make_env, make_vanilla_env, TrackerWrapper, LogWrapper
from tqdm import tqdm
import numpy as np
from purejaxrl.jax_agent import JaxAgent, RawJaxAgent
from rule_based.random.agent import RandomAgent
from rule_based.relicbound.agent import RelicboundAgent
from rule_based.naive.agent import NaiveAgent

def run_match(
        agent_0: Any, 
        agent_1: Any, 
        vanilla_env: TrackerWrapper, 
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

    stack_stats = []

    for step_idx in tqdm(range(max_episode_steps)):
        action = {
            "player_0": agent_0.act(step = step_idx, obs = EnvObs_to_dict(obs["player_0"])), 
            "player_1": agent_1.act(step = step_idx, obs = EnvObs_to_dict(obs["player_1"])),
        }
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done, info = vanilla_env.step(rng, env_state, action, env_params)
        stack_stats.append((info["episode_stats_player_0"], info["episode_stats_player_1"]))

    stats_arrays = {
        "episode_stats_player_0": {stat: np.array([getattr(stack_stats[i][0], stat) for i in range(len(stack_stats))]) for stat in vanilla_env.stats_names},
        "episode_stats_player_1": {stat: np.array([getattr(stack_stats[i][1], stat) for i in range(len(stack_stats))]) for stat in vanilla_env.stats_names}
    }
        
    plot_stats(stats_arrays)


def run_episode_and_record(
        rec_env: LogWrapper, 
        network: Any,
        network_params_0: Any, 
        network_params_1: Any, 
        key: chex.PRNGKey,
        steps: int
    ):
    """
    Evaluate the trained agent against a reference agent using a separate eval environment.
    """
    rng, _rng = jax.random.split(key)

    # sample random params initially
    env_params = sample_params(rng)
    
    # reset 
    rng, _rng = jax.random.split(rng)
    obs, env_state = rec_env.reset(rng, env_params)

    @jax.jit
    def forward(rng, obs, network_params_0, network_params_1):
        # GET OBS BATCHES
        obs_batch_player_0, obs_batch_player_1 = get_obs_batch(obs, rec_env.players)
        obs_batch_player_0 = {feat: jnp.expand_dims(value, axis=0) for feat, value in obs_batch_player_0.items()}
        obs_batch_player_1 = {feat: jnp.expand_dims(value, axis=0) for feat, value in obs_batch_player_1.items()}

        # SELECT ACTION: PLAYER 0
        rng, _rng = jax.random.split(rng)
        logits, value = network.apply(network_params_0, **obs_batch_player_0) # probs is (16, 5)
        action_0 = sample_action(key= _rng, logits=logits)[0] # (16,)

        # SELECT ACTION: PLAYER 1
        rng, _rng = jax.random.split(rng)
        logits, value = network.apply(network_params_1, **obs_batch_player_0) # probs is (16, 5)
        action_1 = sample_action(key= _rng, logits=logits)[0] # (16,)
        return  {rec_env.players[0]: action_0, rec_env.players[1]: action_1}

    stack_stats = []
    
    for _ in tqdm(range(steps)):
        rng, _rng = jax.random.split(rng)
        action = forward(rng, obs, network_params_0, network_params_1)
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done, info = rec_env.step(rng, env_state, action, env_params)
        stack_stats.append((info["episode_stats_player_0"], info["episode_stats_player_1"]))
    
    rec_env.close()

    stats_arrays = {
        "episode_stats_player_0": {stat: np.array([getattr(stack_stats[i][0], stat) for i in range(len(stack_stats))]) for stat in rec_env.stats_names},
        "episode_stats_player_1": {stat: np.array([getattr(stack_stats[i][1], stat) for i in range(len(stack_stats))]) for stat in rec_env.stats_names}
    }
    
    plot_stats(stats_arrays)

def test_a():

    config = parse_config()
    seed = np.random.randint(0, 10000)
    key = jax.random.PRNGKey(seed)
    rec_env = make_env(config["env_args"], record=True, save_on_close=True, save_dir = "test", save_format = "html")
    network = config["network"]["model"]
    rec_env = LogWrapper(rec_env, replace_info=True)
    steps = 100
    
    run_episode_and_record(
        rec_env = rec_env,
        network = network,
        network_params_0 = config["network"]["network_params"],
        network_params_1 = config["network"]["network_params"],
        steps = steps,
        key = key, 
    )



def test_b():

    config = parse_config()
    seed = np.random.randint(0, 10000)
    key = jax.random.PRNGKey(seed)
    vanilla_env = make_vanilla_env(config["env_args"])
    vanilla_env = LogWrapper(vanilla_env)
    env_params = sample_params(key)
    
    run_match(
        agent_0 = NaiveAgent("player_0", env_params.__dict__),
        agent_1 = NaiveAgent("player_1", env_params.__dict__),
        key = key,
        vanilla_env = vanilla_env,
        env_params = env_params
    )


if __name__ == "__main__":
    test_a()
    test_b()