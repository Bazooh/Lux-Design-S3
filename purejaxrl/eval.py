import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agents.base_agent import Agent

import jax, chex
from sample_params import sample_params_fn
import dataclasses
from typing import Any
def eval(
        agent_0_class: Agent, 
        agent_1_class: Agent, 
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
    action_space = (eval_env.action_space())
    sample_action_fn = jax.vmap(action_space.sample)

    # sample random params initially
    rng, _rng = jax.random.split(rng)
    rng_params = jax.random.split(key, num_envs)
    env_params = sample_params_fn(rng_params)
    
    # reset 
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(key, num_envs)
    obs, state = reset_fn(reset_rng, env_params)

    # and make the first step
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(key, num_envs)
    obs, state, reward, done, info = step_fn(
        rng_step,
        state,
        sample_action_fn(rng_step),
        env_params,
    )

    max_episode_steps = (
        eval_env.fixed_env_params.max_steps_in_match + 1
    ) * eval_env.fixed_env_params.match_count_per_episode # 501 * 5 steps per env

    # initialise the agents
    agent_0 = agent_0_class(player = "player_0", env_cfg = dataclasses.asdict(env_params))
    agent_1 = agent_1_class(player = "player_1", env_cfg = dataclasses.asdict(env_params))


    def forward(
        name
    ):
        if name == "player_0":
            return eval_env.action_space()
        else: 
            return eval_env.action_space()

    @jax.jit
    def five_games_rollout(runner_state):
        def _env_step(runner_state, _):
            env_state, last_obs, rng, env_params = runner_state
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, num_envs)

            # forward
            rngs = jax.random.split(_rng, 2*num_envs).reshape((2, num_envs, -1))
            actions = {k: jax.vmap(forward(k).sample)(rngs[i]) for i, k in enumerate(["player_0", "player_1",])}
            
            obs, env_state, reward, done, info = step_fn(
                rng_step,
                env_state,
                actions,
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



if __name__ == "__main__":
    from rule_based.random.agent import Agent
    from make_env import make_env

    seed = 1
    eval_env = make_env(seed)
    eval_key = jax.random.PRNGKey(seed)
    eval(
        agent_0_class = Agent, 
        agent_1_class = Agent, 
        key = eval_key, 
        eval_env = eval_env
    )