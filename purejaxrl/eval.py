import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agents.base_agent import Agent
from network import HybridActorCritic
import jax, chex
import dataclasses
from typing import Any
import jax.numpy as jnp
from utils import sample_action, sample_greedy_action, get_logprob, get_entropy, get_obs_batch, sample_params

def eval_checkpoints(
        network: Any,
        network_params_0: Any, 
        network_params_1: Any, 
        eval_env: Any, 
        key: chex.PRNGKey,
        num_eval_episodes: int = 4
    ):
    """
    Evaluate the trained agent against a reference agent using a separate eval environment.
    """
    num_envs = num_eval_episodes  # the matches are run in different envs
    rng, _rng = jax.random.split(key)

    # define the vmapped functions 
    reset_fn = jax.vmap(eval_env.reset)
    step_fn = jax.vmap(eval_env.step)
    sample_params_fn = jax.vmap(sample_params)

    # sample random params initially
    rng, _rng = jax.random.split(rng)
    rng_params = jax.random.split(key, num_envs)
    env_params = sample_params_fn(rng_params)
    
    # reset 
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(key, num_envs)
    obs, state = reset_fn(reset_rng, env_params)

    max_episode_steps = (
        eval_env.fixed_env_params.max_steps_in_match + 1
    ) * eval_env.fixed_env_params.match_count_per_episode # 101 * 5 steps per env


    @jax.jit
    def five_games_rollout(runner_state):
        def _env_step(runner_state, _):
            env_state, last_obs, rng, env_params = runner_state
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, num_envs)
            
            # GET OBS BATCHES
            last_obs_batch_player_0, last_obs_batch_player_1 = get_obs_batch(last_obs, eval_env.players)

            # SELECT ACTION: PLAYER 0
            rng, _rng = jax.random.split(rng)
            logits, value = network.apply(network_params_0, **last_obs_batch_player_0) # probs is (N, 16, 5)
            action_0 = sample_action(_rng, logits)

            # SELECT ACTION: PLAYER 1
            rng, _rng = jax.random.split(rng)
            logits, value = network.apply(network_params_1, **last_obs_batch_player_0) # probs is (N, 16, 5)
            action_1 = sample_action(_rng, logits)
        
            obs, env_state, reward, done, info = step_fn(
                rng_step,
                env_state,
                {
                    eval_env.players[0]: action_0,
                    eval_env.players[1]: action_1,
                },
                env_params,
            )
            return (env_state, last_obs, rng, env_params), (
                obs,
                env_state,
                reward,
                done,
                info,
            )
        
        _, (obs, env_state, reward, done, info) = jax.lax.scan(
            _env_step, runner_state, length=max_episode_steps, unroll=1
        ) # at the end, reward contains the vector of the different game results
        return reward
    
    runner_state = state, obs, rng, env_params
    reward = five_games_rollout(runner_state)
    return reward


def run_episode_and_record(
        network: Any,
        network_params_0: Any, 
        network_params_1: Any, 
        rec_env: Any, 
        key: chex.PRNGKey,
    ):
    """
    Evaluate the trained agent against a reference agent using a separate eval environment.
    """
    rng, _rng = jax.random.split(key)

    # sample random params initially
    rng, _rng = jax.random.split(rng)
    env_params = sample_params(rng)
    
    # reset 
    rng, _rng = jax.random.split(rng)
    obs, env_state = rec_env.reset(rng, env_params)

    max_episode_steps = (
        eval_env.fixed_env_params.max_steps_in_match + 1
    ) * eval_env.fixed_env_params.match_count_per_episode # 101 * 5 steps per env

    points = jax.numpy.zeros((max_episode_steps, 2))

    @jax.jit
    def jitted_get_actions(rng, obs, network_params_0, network_params_1):
        # GET OBS BATCHES
        obs_batch_player_0, obs_batch_player_1 = get_obs_batch(obs, eval_env.players)
        obs_batch_player_0 = {feat: jnp.expand_dims(value, axis=0) for feat, value in obs_batch_player_0.items()}
        obs_batch_player_1 = {feat: jnp.expand_dims(value, axis=0) for feat, value in obs_batch_player_1.items()}

        # SELECT ACTION: PLAYER 0
        rng, _rng = jax.random.split(rng)
        logits, value = network.apply(network_params_0, **obs_batch_player_0) # probs is (16, 5)
        action_0 = sample_greedy_action(logits)[0] # (16,)

        # SELECT ACTION: PLAYER 1
        rng, _rng = jax.random.split(rng)
        logits, value = network.apply(network_params_1, **obs_batch_player_0) # probs is (16, 5)
        action_1 = sample_greedy_action(logits)[0] # (16,)

        return  {eval_env.players[0]: action_0, eval_env.players[1]: action_1}
    
    for step_idx in range(max_episode_steps):
        rng, _rng = jax.random.split(rng)
        action = jitted_get_actions(rng, obs, network_params_0, network_params_1)
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done, info = rec_env.step(rng, env_state, action, env_params)
        reward_batch =  jnp.stack([reward[a] for a in eval_env.players])

        points = points.at[step_idx].set(reward_batch)

    rec_env.close()
    return points # shape (max_episode_steps, 2)

if __name__ == "__main__":
    from rule_based.random.agent import Agent
    from make_env import make_env

    # EVAL 
    seed = 1
    eval_env = make_env()
    key = jax.random.PRNGKey(seed)
    # INIT NETWORK
    network = HybridActorCritic(
        action_dim=eval_env.action_space.n,
    )
    # init params 0
    rng, _rng = jax.random.split(key)
    init_x = eval_env.observation_space.sample(rng)
    init_x = {feat: jnp.expand_dims(value, axis=0) for feat, value in init_x.items()}
    network_params_0 = network.init(_rng, **init_x)
    rng, _rng = jax.random.split(key)
    network_params_1 = network.init(_rng, **init_x)

    reward = eval_checkpoints(
        network = network,
        network_params_0 = network_params_0,
        network_params_1 = network_params_1,
        key = key, 
        eval_env = eval_env
    )
    print("reward:", reward["player_0"].sum(axis = 0), reward["player_1"].sum(axis = 0))

    # RECORD
    
    rec_env = make_env(record=True, save_dir = "here", save_format = "json")
    key = jax.random.PRNGKey(seed)
    points = run_episode_and_record(
        network = network,
        network_params_0 = network_params_0,
        network_params_1 = network_params_1,
        key = key, 
        rec_env = rec_env
    )
    print("points:", points.sum(axis=0))