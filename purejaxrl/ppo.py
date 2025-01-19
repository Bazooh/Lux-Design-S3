import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax, chex
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from purejaxrl.env.make_env import make_env
from typing import NamedTuple
from jax_tqdm import scan_tqdm
from utils import sample_action, sample_greedy_action, get_logprob, get_entropy, get_obs_batch, init_network_params, sample_params
from purejaxrl.env.wrappers import LogWrapper
from purejaxrl.parse_config import parse_config
import wandb
from purejaxrl.eval import run_episode_and_record
"""
Reference:  PPO implementation from PUREJAXRL
https://github.com/Hadrien-Cr/purejaxrl/blob/main/purejaxrl/ppo.py

I changed it to feature 2 players, represented by two 'network_params':
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


def make_train(config, debug=False,):
        
    config["ppo"]["num_updates"] =  (config["ppo"]["total_timesteps"]// (config["ppo"]["num_steps"] * config["ppo"]["num_envs"]))
    config["ppo"]["minibatch_size"] = (config["ppo"]["num_envs"] * config["ppo"]["num_steps"] // config["ppo"]["num_minibatches"])
    print('-'*100)
    print("Starting PPO with config:", config["ppo"])
    print('-'*100)
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["ppo"]["num_minibatches"] * config["ppo"]["update_epochs"]))
            / config["ppo"]["num_updates"]
        )
        return config["ppo"]["lr"] * frac

    env = make_env(config["env_args"])
    env = LogWrapper(env, replace_info=True)
    model = config["network"]["model"]

    def train(key: chex.PRNGKey,):
        
        # INITIALIZE NETWORK
        rng, _rng = jax.random.split(key)
        network_params_0 = init_network_params(_rng, model, init_x=env.observation_space.sample(_rng))
        rng, _rng = jax.random.split(key)
        network_params_1 = init_network_params(_rng, model, init_x=env.observation_space.sample(_rng))

        #create TrainState objects
        transform_lr = optax.chain(
            optax.clip_by_global_norm(config["ppo"]["clip_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=model.apply,
            params=network_params_0,
            tx=transform_lr,
        )

        # INITIALIZE ENVIRONMENT
        # sample random params initially
        rng, _rng = jax.random.split(rng)
        rng_params = jax.random.split(key, config["ppo"]["num_envs"])
        env_params = jax.vmap(lambda key: sample_params(key, match_count_per_episode=1))(rng_params)
        
        # reset 
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(key, config["ppo"]["num_envs"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng, env_params)

        # TRAIN LOOP
        @scan_tqdm(config["ppo"]["num_updates"], print_rate=1)
        def _update_step(runner_state, update_i):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, update_i):
                train_state, network_params_1, env_state, last_obs, rng, env_params = runner_state
                #jax.debug.print("unit_sap_range: {}", env_params.unit_sap_range)
                
                # GET OBS BATCHES
                last_obs_batch_player_0, last_obs_batch_player_1 = get_obs_batch(last_obs, env.players)

                # SELECT ACTION: PLAYER 0
                rng, _rng = jax.random.split(rng)
                logits, value = model.apply(train_state.params, **last_obs_batch_player_0) # probs is (N, 16, 6)
                mask_awake = (last_obs_batch_player_0['position'][..., 0] >= 0).astype(jnp.float32)  # Shape: (N, 16), 1 if position >= 0 else 0
                action_0 = sample_action(_rng, logits, noise_std=config["ppo"]["action_noise"])
                log_prob_0 = get_logprob(logits, mask_awake, action_0)

                # SELECT ACTION: PLAYER 1
                rng, _rng = jax.random.split(rng)
                logits, value = model.apply(train_state.params, **last_obs_batch_player_0) # probs is (N, 16, 6)
                mask_awake = (last_obs_batch_player_1['position'][..., 0] >= 0).astype(jnp.float32)  # Shape: (N, 16), 1 if position >= 0 else 0
                action_1 = sample_action(_rng, logits, noise_std=config["ppo"]["action_noise"])
                log_prob_1 = get_logprob(logits, mask_awake, action_1)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["ppo"]["num_envs"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(rng_step, env_state, {env.players[0]: action_0, env.players[1]: action_1}, env_params)
                reward_batch =  jnp.stack([reward[a] for a in env.players])
                reward_batch_player_0 = reward_batch[0]
                reward_batch_player_1 = reward_batch[1]

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
                _env_step, runner_state, None, config["ppo"]["num_steps"],
            )

            # CALCULATE ADVANTAGE
            train_state, network_params_1, env_state, last_obs, rng, env_params = runner_state
            # GET OBS BATCHES
            last_obs_batch_player_0, last_obs_batch_player_1 = get_obs_batch(last_obs, env.players)
            _, last_val = model.apply(train_state.params, **last_obs_batch_player_0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["ppo"]["gamma"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["ppo"]["gamma"] * config["ppo"]["gae_lambda"] * (1 - done) * gae
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
                        logits, value = model.apply(params, **traj_batch.obs)
                        mask_awake = (traj_batch.obs['position'][..., 0] >= 0).astype(jnp.float32)
                        log_prob = get_logprob(logits, mask_awake, traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["ppo"]["clip_eps"], config["ppo"]["clip_eps"])
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
                                1.0 - config["ppo"]["clip_eps"],
                                1.0 + config["ppo"]["clip_eps"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = get_entropy(logits).mean()

                        total_loss = (
                            loss_actor
                            + config["ppo"]["vf_coef"] * value_loss
                            - config["ppo"]["ent_coef"] * entropy
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
                batch_size = config["ppo"]["minibatch_size"] * config["ppo"]["num_minibatches"]
                assert (
                    batch_size == config["ppo"]["num_steps"] * config["ppo"]["num_envs"]
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
                        x, [config["ppo"]["num_minibatches"], -1] + list(x.shape[1:])
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
                _update_epoch, update_state, None, config["ppo"]["update_epochs"]
            )
            train_state = update_state[0]
            game_info = traj_batch.info
            rng = update_state[-1]
            
            # Debugging mode
            if debug:
                def callback(metric):
                    game_info, loss_info = metric
                    return_values = jnp.mean(game_info["episode_return"][game_info["returned_episode"]], axis = 0)
                    return_points = jnp.mean(game_info["episode_points"][game_info["returned_episode"]], axis = 0)
                    episode_wins = jnp.mean(game_info["episode_wins"][game_info["returned_episode"]], axis = 0)
                    episode_winner = jnp.mean(game_info["episode_winner"][game_info["returned_episode"]], axis = 0)
                    timesteps = game_info["global_timestep"][game_info["returned_episode"]] * config["ppo"]["num_envs"]
                    global_timestep = jnp.sum(timesteps)
                    metrics = {
                        "return_values": return_values[0],
                        "return_points": return_points[0],
                        "episode_wins": episode_wins[0],
                        "episode_winner": episode_winner[0],
                    }

                    if len(timesteps) > 0: 
                        wandb.log(metrics)
                        print(
                            f"timesteps: {global_timestep}, return_values: {return_values[0]:.2f}, return_points: {return_points[0]:.2f}, episode_wins: {episode_wins[0]:.2f}, episode_winner: {episode_winner[0]:.2f}"
                        )
                jax.debug.callback(callback, (game_info, loss_info))          
            runner_state = (train_state, network_params_1, env_state, last_obs, rng, env_params)
            
            return runner_state, game_info

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, network_params_1, env_state, obsv, _rng, env_params)
        runner_state, game_info = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["ppo"]["num_updates"]),
        )
        return {"runner_state": runner_state, "metrics": game_info}
    

    return train


if __name__ == "__main__":
    wandb.init(
        project = "LuxAIS3",
        name = "purejaxrl_ppo",
        mode = "online",
    )
    jax.config.update("jax_numpy_dtype_promotion", "standard")
    config = parse_config()
    train_jit = jax.jit(make_train(config, debug=True))
    rng = jax.random.PRNGKey(seed = config["ppo"]["seed"])
    out = train_jit(rng)
    wandb.finish()
    
    runner_state = out["runner_state"]
    train_state = runner_state[0]
    rec_env = make_env(config["env_args"], record=True, save_on_close=True, save_dir = "purejaxrl/ppo_replays", save_format = "html")
    run_episode_and_record(
        rec_env=rec_env,
        network=config["network"]["model"],
        network_params_0=train_state.params,
        network_params_1=train_state.params,
        key=rng,
    )