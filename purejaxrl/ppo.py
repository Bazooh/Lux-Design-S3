import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax, chex
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from make_env import make_env
from typing import NamedTuple, Any
import time
from jax_tqdm import scan_tqdm
from utils import sample_action, sample_greedy_action, get_logprob, get_entropy, get_obs_batch, init_network_params, sample_params
from purejaxrl.wrappers.base_wrappers import LogWrapper
from purejaxrl.parse_config import parse_config
"""
Reference:  PPO implementation from PUREJAXRL
https://github.com/Hadrien-Cr/purejaxrl/blob/main/purejaxrl/ppo.py

I changed it to feature 2 players, represented by 'two network_params':
    - the first player is learning
    - the second is not learning but is overwritten by player 1 every few games
"""
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: dict
    info: jnp.ndarray


def make_train(
        config_path: str = "purejaxrl/jax_config.yaml",
        total_timesteps: int = 1e5,
        num_steps: int = 128,
        lr_start: float = 2.5e-4,
        num_envs: int = 4,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coeff: float = 0.5,
        clip_grad_norm: float = 0.5,
        freq_opponent_updates: int = 20,
        seed: int = 0,
        debug: bool = True
):
    num_updates=  (
       total_timesteps// (num_steps * num_envs)
    )
    minibatch_size = (
        num_envs * num_steps // num_minibatches
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (num_minibatches * update_epochs))
            / num_updates
        )
        return lr_start * frac

    env = make_env(config_path)
    env = LogWrapper(env)
    config = parse_config(config_path)
    
    def train(key: chex.PRNGKey,):
        start_time = time.time()
        
        # INIT NETWORK
        network = config["network"]["network"]
        # init params 0
        rng, _rng = jax.random.split(key)
        network_params_0 = init_network_params(_rng, network, init_x=env.observation_space.sample(_rng))
        rng, _rng = jax.random.split(key)
        network_params_1 = init_network_params(_rng, network, init_x=env.observation_space.sample(_rng))

        #create TrainState objects
        transform_lr = optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params_0,
            tx=transform_lr,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs) # create num_envs seed out of 1 seed
        reset_fn = jax.vmap(env.reset)
        step_fn = jax.vmap(env.step)
        sample_params_fn = jax.vmap(sample_params)
        
        # sample random params initially
        rng, _rng = jax.random.split(rng)
        rng_params = jax.random.split(key, num_envs)
        env_params = sample_params_fn(rng_params)
        
        # reset 
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(key, num_envs)
        obsv, env_state = reset_fn(reset_rng, env_params)


        
        # TRAIN LOOP
        @scan_tqdm(num_updates)
        def _update_step(runner_state, update_i):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, update_i):
                train_state, network_params_1, env_state, last_obs, rng, env_params = runner_state
                #jax.debug.print("unit_sap_range: {}", env_params.unit_sap_range)
                
                # GET OBS BATCHES
                last_obs_batch_player_0, last_obs_batch_player_1 = get_obs_batch(last_obs, env.players)

                # SELECT ACTION: PLAYER 0
                rng, _rng = jax.random.split(rng)
                logits, value = network.apply(train_state.params, **last_obs_batch_player_0) # probs is (N, 16, 6)
                mask_awake = (last_obs_batch_player_0['position'][..., 0] >= 0).astype(jnp.float32)  # Shape: (N, 16), 1 if position >= 0 else 0
                action_0 = sample_action(_rng, logits)
                log_prob_0 = get_logprob(logits, mask_awake, action_0)

                # SELECT ACTION: PLAYER 1
                rng, _rng = jax.random.split(rng)
                logits, value = network.apply(train_state.params, **last_obs_batch_player_0) # probs is (N, 16, 6)
                mask_awake = (last_obs_batch_player_1['position'][..., 0] >= 0).astype(jnp.float32)  # Shape: (N, 16), 1 if position >= 0 else 0
                action_1 = sample_action(_rng, logits)
                log_prob_1 = get_logprob(logits, mask_awake, action_1)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                obsv, env_state, reward, done, info = step_fn(rng_step, env_state, {env.players[0]: action_0, env.players[1]: action_1}, env_params)
                reward_batch =  jnp.stack([reward[a] for a in env.players])
                reward_batch_player_0 = reward_batch[0]
                reward_batch_player_1 = reward_batch[1]

                # Reset environments where `done` is True
                def reset_or_keep(obs, state, params, done, rng):
                    def end_game_single():
                        reset_env_params = sample_params(rng)
                        obs_re, state_re = env.reset(rng, reset_env_params)
                        return obs_re, state_re, reset_env_params

                    obs, state, params = jax.lax.cond(
                        done,
                        end_game_single,
                        lambda: (obs, state, params),
                    )
                    return obs, state, params

                # Vectorize reset logic
                obsv, env_state, env_params = jax.vmap(reset_or_keep)(obsv, env_state, env_params, done, rng_step)

                transition = Transition(
                    done = done,
                    action = action_0, 
                    value = value, 
                    reward = reward_batch_player_0, 
                    log_prob = log_prob_0, 
                    obs = last_obs_batch_player_0, 
                    info = info
                )
                runner_state = (train_state, network_params_1, env_state, obsv, rng, env_params)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, network_params_1, env_state, last_obs, rng, env_params = runner_state
            # GET OBS BATCHES
            last_obs_batch_player_0, last_obs_batch_player_1 = get_obs_batch(last_obs, env.players)
            _, last_val = network.apply(train_state.params, **last_obs_batch_player_0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + gamma * next_value * (1 - done) - value
                    gae = (
                        delta
                        + gamma * gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        logits, value = network.apply(params, **traj_batch.obs)
                        mask_awake = (traj_batch.obs['position'][..., 0] >= 0).astype(jnp.float32)
                        log_prob = get_logprob(logits, mask_awake, traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-clip_eps, clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - clip_eps,
                                1.0 + clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = get_entropy(logits).mean()

                        total_loss = (
                            loss_actor
                            + vf_coeff * value_loss
                            - ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)


                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Batching and Shuffling
                batch_size = minibatch_size * num_minibatches
                assert (
                    batch_size == num_steps * num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)

                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, update_epochs
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # Debugging mode
            if debug:
                def callback(info):
                    timesteps = info["timestep"][info["returned_episode"]] * num_envs
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, fps={(num_envs*num_steps*num_steps)/(time.time() - start_time):.2f}")
                jax.debug.callback(callback, metric)
                
            runner_state = (train_state, network_params_1, env_state, last_obs, rng, env_params)
            start_time = time.time()

            # Overwrite the player 1 weights every freq_opponent_updates
            runner_state = jax.lax.cond(
                (update_i % freq_opponent_updates) == 0,
                lambda runner_state: (runner_state[0], 
                                    runner_state[0].params, # overwriting network_params_1 with network_params_0
                                    runner_state[2], 
                                    runner_state[3], 
                                    runner_state[4],
                                    runner_state[5]), 
                lambda runner_state: runner_state, # skip
                runner_state,
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, network_params_1, env_state, obsv, _rng, env_params)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_updates)
        )
        return {"runner_state": runner_state, "metrics": metric}
    

    return train


if __name__ == "__main__":
    jax.config.update("jax_numpy_dtype_promotion", "standard")
    args = {
        "total_timesteps": 1e5,
        "num_envs": 8,
        "debug": False,
    }

    train_jit = jax.jit(make_train(**args))
    rng = jax.random.PRNGKey(seed = 0)
    out = train_jit(rng)