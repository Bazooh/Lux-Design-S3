import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from luxai_s3.env import LuxAIS3Env
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from tqdm import tqdm
import jax, chex
import jax.numpy as jnp
import optax
from purejaxrl.env.make_env import make_env, make_vanilla_env, LogWrapper
from purejaxrl.env.wrappers import RewardObject, gamma_from_reward_phase
from typing import NamedTuple
from jax_tqdm import scan_tqdm
from purejaxrl.utils import (
    sample_group_action,
    get_logprob, 
    get_entropy, 
    get_obs_batch, 
    save_state_dict, 
    create_checkpoint_manager,
    CustomTrainState, 
)
from purejaxrl.env.utils import sample_params
from purejaxrl.parse_config import parse_config
import wandb
from purejaxrl.eval_jax import run_arena_jax_agents, run_episode_and_record
from purejaxrl.eval_standard import run_arena_standard_agents
from purejaxrl.purejaxrl_agent import RawPureJaxRLAgent
from datetime import datetime
from functools import partial
"""
Reference:  ppo implementation from PUREJAXRL
https://github.com/Hadrien-Cr/purejaxrl/blob/main/purejaxrl/ppo.py

I changed it to feature 2 players, represented by two 'network_params':
    - the first player is learning
    - the second is not learning but is overwritten by player 1 every few games
"""
class Transition(NamedTuple):
    done_v: jnp.ndarray
    action_v: jnp.ndarray
    value_v: jnp.ndarray
    reward_v: jnp.ndarray
    log_prob_v: jnp.ndarray
    obs_v: dict
    info_v: jnp.ndarray


@partial(jax.jit, static_argnums=(3))
def compute_reshaped_reward(start_phase: int, weight: float, rewards: chex.Array, reward_smoothing: bool = True):
    """
    Adjust the gamma depending on the curriculum phase
    start_phase: the current phase
    weight: the weight between the current phase and the next phase in (0, 1)
    rewards: Array of shape (N, n_phases)
    reward_smoothing: whether to smooth reward between phases
    """
    reshaped_reward = jax.lax.select(
        start_phase == len(rewards),
        rewards[:,-1],
        jax.lax.select(
            reward_smoothing,
            rewards[:,start_phase] * (1 - weight) + rewards[:,start_phase + 1] * weight,
            rewards[:,start_phase]
        ),
    )
    return reshaped_reward

def make_train(config, debug=False,):
    if os.path.exists(config["ppo"]["save_checkpoint_path"]):
        config["ppo"]["save_checkpoint_path"] += datetime.now().strftime("_%H_%M")

    config["ppo"]["num_updates"] =  (config["ppo"]["total_timesteps"]// (config["ppo"]["num_steps"] * config["ppo"]["num_envs"]))
    config["ppo"]["minibatch_size"] = (config["ppo"]["num_envs"] * config["ppo"]["num_steps"] // config["ppo"]["num_minibatches"])
    
    print('-'*150)
    print("Network has args: ", { k: v for k, v in config["network_args"].items()}, "\n", '-'*150)
    print("Reward phases:", config["env_args"]["reward_phases"],"\n", '-'*150)
    print("Starting ppo with config:", config["ppo"], "\n",'-'*150)

    if config["arena_jax"] is not None:
        print("Arena jax opponent is:", config["arena_jax"]["agent"].__name__, "; arena freq is:", config["arena_jax"]["arena_freq"], "\n",'-'*150)
    if config["arena_std"] is not None:
        print("Arena jax opponent is:", config["arena_std"]["agent"].__name__, "; arena freq is:", config["arena_std"]["arena_freq"], "\n",'-'*150)
  
    checkpoint_manager = create_checkpoint_manager(config["ppo"]["save_checkpoint_path"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["ppo"]["num_minibatches"] * config["ppo"]["update_epochs"]))
            / config["ppo"]["num_updates"]
        )
        return config["ppo"]["lr"] * frac

    env = make_env(config["env_args"])
    env = LogWrapper(env, replace_info=True)    
    start_state_dict = config["network"]["state_dict"]
    model = config["network"]["model"]
    num_phases = env.num_phases
    
    if config["arena_jax"] is not None:
        arena_jax_agent = config["arena_jax"]["agent"]
        rec_env_jax = make_vanilla_env(config["env_args"], record=True, save_on_close=True, save_dir = "replays_ppo_jax", save_format = "html")
        rec_env_jax = LogWrapper(rec_env_jax)
        arena_env_jax = make_vanilla_env(config["env_args"])
        arena_env_jax = LogWrapper(arena_env_jax)    
        
    if config["arena_std"] is not None:
        arena_std_agent = config["arena_std"]["agent"]
        rec_env_gym = RecordEpisode(LuxAIS3GymEnv(numpy_output=True), save_on_close=True, save_dir = "replays_ppo_std")
        arena_env_gym = LuxAIS3GymEnv(numpy_output=True)

    reward_phases = env.reward_phases

    @partial(jax.jit, static_argnums=(3))
    def compute_reshaped_gamma(start_phase: int, weight: float, gamma: float, gamma_smoothing: bool = True):
        """
        Adjust the gamma depending on the curriculum phase
        """
        gammas = jnp.array([gamma_from_reward_phase(reward_phase=reward_phase, gamma = gamma) for reward_phase in reward_phases])
        reshaped_gamma = jax.lax.select(
            start_phase == len(reward_phases),
            gammas[-1],
            jax.lax.select(
                gamma_smoothing,
                gammas[start_phase]* (1 - weight) + gammas[start_phase+1] * weight,
                gammas[start_phase]            
            ),
        )
        return reshaped_gamma  


    def train(rng: chex.PRNGKey,):

        #create TrainState objects
        transform_lr = optax.chain(
            optax.clip_by_global_norm(config["ppo"]["clip_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )

        train_state = CustomTrainState.create(
            apply_fn = model.apply,
            params = start_state_dict["params"],
            batch_stats = start_state_dict["batch_stats"],
            tx=transform_lr,
        )
        opp_state_dict = start_state_dict

        @jax.jit
        def step_and_keep_or_reset(rng, env_state, actions, env_params):
            obs, next_env_state, reward, done, info = env.step(rng, env_state, actions, env_params)
            
            def reset():
                reset_rng, param_rng = jax.random.split(rng)
                new_params = sample_params(param_rng, match_count_per_episode=config["ppo"]["match_count_per_episode"])
                new_obs, new_state = env.reset(reset_rng, new_params)
                return new_obs, new_state, new_params
            
            # Keep current state or reset based on done flag
            obs, env_state, env_params = jax.lax.cond(
                done,
                reset,
                lambda: (obs, next_env_state, env_params)
            )
            return obs, env_state, reward, done, info, env_params
        
        # INITIALIZE ENVIRONMENT
        # sample random params initially
        rng, _rng = jax.random.split(rng)
        rng_params = jax.random.split(rng, config["ppo"]["num_envs"])
        env_params = jax.vmap(lambda key: sample_params(key, match_count_per_episode = config["ppo"]["match_count_per_episode"]))(rng_params)
        
        # reset 
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(rng, config["ppo"]["num_envs"])
        obs_v, env_state_v = jax.vmap(env.reset)(reset_rng, env_params)

        # TRAIN LOOP
        @scan_tqdm(config["ppo"]["num_updates"], print_rate=1, desc = "Training PPO")
        def _update_step(runner_state, update_i):
            ################# STEP IN THE ENVIRONMENT ADVANTAGE #################
            def _env_step(runner_state, _):
                # GET OBS BATCHES
                train_state, opp_state_dict, env_state_v, last_obs_v, rng, env_params = runner_state
                last_obs_batch_player_0, last_obs_batch_player_1 = get_obs_batch(last_obs_v, env.agents)

                # SELECT ACTION: PLAYER 0
                rng, _rng = jax.random.split(rng)
                rng_v = jax.random.split(_rng, config["ppo"]["num_envs"])
                logits0_v, value0_v, _  = model.apply({"params": train_state.params, "batch_stats": train_state.batch_stats}, **last_obs_batch_player_0) # probs is (N, 16, 6)

                mask_awake0_v = last_obs_batch_player_0['mask_awake'].astype(jnp.float32) # mask is (N, 16)
                action_mask0_v = last_obs_batch_player_0['action_mask'].astype(jnp.float32)
                action0_v = jax.vmap(sample_group_action, in_axes=(0, 0, 0, None))(rng_v, logits0_v, action_mask0_v, config["ppo"]["action_temperature"]) # action is (N, 16)
                log_prob0_v = jax.vmap(get_logprob)(logits0_v, mask_awake0_v, action0_v, action_mask0_v)

                # SELECT ACTION: PLAYER 1
                rng, _rng = jax.random.split(rng)
                rng_v = jax.random.split(_rng, config["ppo"]["num_envs"])
                action_mask1_v = last_obs_batch_player_1['action_mask'].astype(jnp.float32)
                logits1_v, _, _  = model.apply(opp_state_dict, **last_obs_batch_player_1) # logits is (N, 16, 6)
                action1_v = jax.vmap(sample_group_action, in_axes=(0, 0, 0, None))(rng_v, logits1_v, action_mask1_v, config["ppo"]["action_temperature"]) # action is (N, 16)

                # STEP THE ENVIRONMENT
                rng, _rng = jax.random.split(rng)
                rng_v = jax.random.split(_rng, config["ppo"]["num_envs"])
                obs_v, env_state_v, reward_v, done_v, info_v, env_params = jax.vmap(step_and_keep_or_reset)(
                    rng_v, 
                    env_state_v, 
                    {"player_0": action0_v, "player_1": action1_v},
                    env_params
                )

                # LOG THE TRANSITION                 
                transition = Transition(
                    done_v = done_v,
                    action_v = action0_v, 
                    value_v = value0_v, 
                    reward_v = reward_v["player_0"], 
                    log_prob_v = log_prob0_v, 
                    obs_v = last_obs_batch_player_0, 
                    info_v = info_v
                )

                runner_state = (train_state, opp_state_dict, env_state_v, obs_v, rng, env_params)
                return runner_state, transition
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ppo"]["num_steps"],
            )
            train_state, opp_state_dict, env_state_v, last_obs_v, rng, env_params = runner_state

            ################# Self-play opponent update #################
            update_condition = (update_i % config["ppo"]["selfplay_freq_update"]) == 0
            opp_state_dict = {
                "params": jax.tree.map(lambda x, y: jax.lax.select(update_condition, x, y),
                                    train_state.params, opp_state_dict["params"]),
                "batch_stats": jax.tree.map(lambda x, y: jax.lax.select(update_condition, x, y),
                                        train_state.batch_stats, opp_state_dict["batch_stats"]),
            }

            ################# RESHAPING R & GAMMA #####################
            frac = update_i/ config["ppo"]["num_updates"]
            start_phase = jnp.round(frac * num_phases).astype(int)
            weight = (frac * num_phases) - start_phase
            
            reshaped_reward_v = jax.vmap(compute_reshaped_reward, in_axes=(None, None, 0, None))(
                start_phase, 
                weight,
                traj_batch.reward_v,
                env.reward_smoothing 
            )

            reshaped_gamma = compute_reshaped_gamma(
                start_phase, 
                weight,
                gamma = config["ppo"]["gamma"],
                gamma_smoothing = config['ppo']['gamma_smoothing'],
            )  
        
            # jax.debug.print("start phase: {s}, weight: {w}, tg = {tg}, r = {r}, tr = {tr}", 
            #     s = start_phase, 
            #     w = weight, 
            #     r = jnp.mean(traj_batch.reward_v, axis = (0,1)), 
            #     tr = jnp.mean(reshaped_reward_v), 
            #     tg = reshaped_gamma
            # )
            traj_batch = traj_batch._replace(reward_v = reshaped_reward_v)

            ################# CALCULATE ADVANTAGE OVER THE LAST TRAJECTORIES #################
            last_obs_batch_player_0, _ = get_obs_batch(last_obs_v, env.agents) # GET OBS BATCHES
            _, last_val_v,_ = model.apply({"params": train_state.params, "batch_stats": train_state.batch_stats}, **last_obs_batch_player_0) # COMPUTE VALUES

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition: Transition):

                    gae, next_value = gae_and_next_value
                    done_v, value_v, reward_v = (
                        transition.done_v,
                        transition.value_v,
                        transition.reward_v,
                    )
                    delta = reward_v + reshaped_gamma * next_value * (1 - done_v) - value_v
                    gae = (
                        delta
                        + reshaped_gamma * config["ppo"]["gae_lambda"] * (1 - done_v) * gae
                    )
                    return (gae, value_v), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value_v

            advantages, targets = _calculate_gae(traj_batch, last_val_v)

            
            ################# POLICY + CRITIC UPDATE, PERFORMED BY MINIBATCHES #################
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # Apply the model with batch_stats and mutable updates
                        (logits0_v, value0_v, _), updates = train_state.apply_fn(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            **traj_batch.obs_v,
                            train=True,
                            mutable=["batch_stats"],
                        )
                        mask_awake0_v = traj_batch.obs_v["mask_awake"]
                        action_mask_v = traj_batch.obs_v["action_mask"]
                        log_prob0_v = jax.vmap(get_logprob)(logits0_v, mask_awake0_v, traj_batch.action_v, action_mask_v)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value_v + (
                            value0_v - traj_batch.value_v
                        ).clip(-config["ppo"]["clip_eps"], config["ppo"]["clip_eps"])
                        value_losses = jnp.square(value0_v - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        
                        explained_var = jax.lax.select(
                            jnp.var(targets) > 0,
                            1.0 - jnp.var(value0_v - targets) / jnp.var(targets),
                            0.0
                        )
                        
                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob0_v - traj_batch.log_prob_v)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        actor_loss1 = ratio * gae
                        actor_loss2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["ppo"]["clip_eps"],
                                1.0 + config["ppo"]["clip_eps"],
                            )
                            * gae
                        )
                        
                        clip_frac = (actor_loss1 > actor_loss2).mean()
                        
                        actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
                        actor_loss = actor_loss.mean()
                        entropy = jax.vmap(get_entropy)(logits0_v, mask_awake0_v, action_mask_v).mean()

                        total_loss = (
                            actor_loss
                            + config["ppo"]["vf_coef"] * value_loss
                            - config["ppo"]["ent_coef"] * entropy
                        )

                        return total_loss, {
                            "value_loss": value_loss,
                            "actor_loss": actor_loss,
                            "entropy": entropy,
                            "clip_frac": clip_frac,
                            "batch_stats": updates["batch_stats"],
                            "explained_var": explained_var
                        }

                    # Compute gradients and update train state
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                    # Apply gradients and update batch_stats
                    train_state = train_state.apply_gradients(
                        grads=grads, batch_stats=aux["batch_stats"]
                    )

                    return train_state, (total_loss, {
                            "value_loss":  aux["value_loss"],
                            "actor_loss":  aux["actor_loss"],
                            "entropy": aux["entropy"],
                            "clip_frac": aux["clip_frac"],   
                            "explained_var": aux["explained_var"]                         
                        })
                
                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Batching and Shuffling
                batch_size = config["ppo"]["minibatch_size"] * config["ppo"]["num_minibatches"]
                assert (batch_size == config["ppo"]["num_steps"] * config["ppo"]["num_envs"]), "batch size must be equal to number of steps * number of envs"
                    
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
            
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["ppo"]["update_epochs"]
            )
            train_state = update_state[0]
            game_info = traj_batch.info_v
            rng = update_state[-1]
            (loss, loss_dict) = loss_info


            ################# LOG COOL STUFF #################
            def callback(game_info, loss, loss_dict, update_i, current_state_dict):
                update_step = int(update_i)

                rng_callback = jax.random.PRNGKey(update_step)

                ########### SAVE CHECKPOINT ###########
                if update_step % config['ppo']["save_checkpoint_freq"] == 0 or update_step == config['ppo']['num_updates'] - 1: 
                    save_state_dict(current_state_dict, checkpoint_manager, step=update_step//config['ppo']['save_checkpoint_freq'])
                
                ############# COMPUTE GAME METRICS ############ 
                returned_episodes = game_info["returned_episode"]
                metrics = {}

                if jnp.sum(returned_episodes) > 0:

                    # Losses
                    total_loss = jnp.mean(loss)
                    value_loss = jnp.mean(loss_dict["value_loss"])
                    actor_loss = jnp.mean(loss_dict["actor_loss"])
                    entropy = jnp.mean(loss_dict["entropy"]) 
                    clip_frac = jnp.mean(loss_dict["clip_frac"])
                    explained_var = jnp.mean(loss_dict["explained_var"])
                    metrics = {
                        "loss/total_loss": total_loss,
                        "loss/value_loss": value_loss,
                        "loss/actor_loss": actor_loss,
                        "loss/entropy": entropy,
                        "loss/clip_frac": clip_frac,
                        "loss/explained_var": explained_var                    
                    }

                    # Rewards
                    return_values_v = game_info["episode_return_player_0"][returned_episodes]
                    return_values_per_phases = jnp.mean(return_values_v, axis=0)     
                      
                    for i in range(num_phases): 
                        metrics[f"reward/selfplay_return_phase_{i}"] = return_values_per_phases[i]
                    frac = update_step/ config["ppo"]["num_updates"]
                    
                    start_phase = jnp.round(frac * num_phases).astype(int)
                    weight = (frac * num_phases) - start_phase
                    reshaped_return_values_v = compute_reshaped_reward(start_phase, weight, return_values_v, reward_smoothing=env.reward_smoothing)
                    reshaped_return_values = jnp.mean(reshaped_return_values_v, axis=0)
                    metrics["reward/selfplay_reshaped_return"] = reshaped_return_values


                    # Stats
                    player_stats = game_info["episode_stats_player_0"].__dict__
                    for key, value in player_stats.items():
                        metrics[f"reward/selfplay_{key}"] = jnp.mean(
                            value[returned_episodes], axis=0
                        )
                    winrate = jnp.mean(player_stats["wins"][returned_episodes] > 0, axis=0)
                    metrics["reward/selfplay_winrate"] = winrate

                    if config['ppo']['verbose'] > 0:
                        print(
                            "------------------------------------\n"  
                            + f"| Return Reshaped     | {reshaped_return_values:<10.4f} |\n"
                            + "\n".join(
                            [f"| Return Phase {i:<6} | {return_values_per_phases[i]:<10.4f} |" for i in range(num_phases)]
                        )
                        + "\n"
                        + (
                            "------------------------------------\n"
                            f"| Update Step         | {update_step:<10d} |\n"
                            f"| Win Rate            | {100 * winrate:<7.1f} %  |\n"
                            f"| Entropy             | {entropy:<10.4f} |\n"
                            f"| Actor Loss          | {actor_loss:<10.4f} |\n"
                            f"| Value Loss          | {value_loss:<10.4f} |\n"
                            f"| Clip Frac           | {clip_frac:<10.4f} |\n"
                            "------------------------------------"
                        ))


                ############ RUN ARENA JAX ############
                if config["arena_jax"] is not None:
                    ############ MATCHES ############
                    if update_step % config['arena_jax']["arena_freq"] == 0 and update_step > 0:
                        our_agent = RawPureJaxRLAgent(
                            player="player_0",
                            model = model,
                            transform_action=config["env_args"]["transform_action"],
                            transform_obs=config["env_args"]["transform_obs"],
                            state_dict=current_state_dict,
                            memory=config["env_args"]["memory"],
                        )

                        arena_info = run_arena_jax_agents(
                            agent_0=our_agent,
                            agent_1=arena_jax_agent("player_1"),
                            vanilla_env=arena_env_jax,
                            key = rng_callback,
                            number_of_games = config["arena_jax"]["num_matches"],
                        )
                        arena_stats = arena_info["episode_stats_player_0"].__dict__
                        arena_winrate = jnp.mean(arena_stats["wins"][arena_info["returned_episode"]] > 0, axis=0)
                        metrics["reward/arena_jax_winrate"] = arena_winrate

                    if update_step % config['arena_jax']["record_freq"] == 0 and update_step > 0:
                        our_agent = RawPureJaxRLAgent(
                            player="player_0",
                            model = model,
                            transform_action=env.transform_action,
                            transform_obs=env.transform_obs,
                            state_dict=current_state_dict,
                            memory=env.memory,
                        )
                        run_episode_and_record(
                            rec_env=rec_env_jax,                        
                            agent_0=our_agent,
                            agent_1=arena_jax_agent("player_1"),
                            key=rng_callback,
                        )
                    
                ############ RUN ARENA STD ############
                if config["arena_std"] is not None:
                    if update_step % config['arena_std']["arena_freq"] == 0 and update_step > 0:
                        our_agent = RawPureJaxRLAgent(
                            player="player_0",
                            model = model,
                            transform_action=config["env_args"]["transform_action"],
                            transform_obs=config["env_args"]["transform_obs"],
                            state_dict=current_state_dict,
                            memory=config["env_args"]["memory"],
                        )
                        arena_winrate = 0
                        for _ in tqdm(range(config["arena_std"]["num_matches"])):
                            reward=run_arena_standard_agents(
                                agent_0_instantiator=lambda: our_agent,
                                agent_1_instantiator=lambda env_params: arena_std_agent("player_1", env_params),
                                gym_env = arena_env_gym,
                                use_tdqm=True
                            )
                            arena_winrate += (1 if reward["player_0"] > reward["player_0"] else 0) / config["arena_std"]["num_matches"]
                        
                        metrics["reward/arena_std_winrate"] = arena_winrate

                    if update_step % config['arena_std']["record_freq"] == 0 and update_step > 0:
                        our_agent = RawPureJaxRLAgent(
                            player="player_0",
                            model = model,
                            transform_action=config["env_args"]["transform_action"],
                            transform_obs=config["env_args"]["transform_obs"],
                            state_dict=current_state_dict,
                            memory=config["env_args"]["memory"],
                        )
                        run_arena_standard_agents(
                            agent_0_instantiator=lambda: our_agent,
                            agent_1_instantiator=lambda env_params: arena_std_agent("player_1", env_params),
                            gym_env = rec_env_gym,
                            use_tdqm=True
                        )
                        rec_env_gym.close()

                
                if config["ppo"]["use_wandb"]: wandb.log(metrics, step=update_step*config["ppo"]["num_envs"]*config["ppo"]["num_steps"])

            if debug: 
                jax.debug.callback(callback, game_info, loss, loss_dict, update_i, {"params": train_state.params, "batch_stats": train_state.batch_stats})

            runner_state = (train_state, opp_state_dict, env_state_v, last_obs_v, rng, env_params)
            
            return runner_state, game_info

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, opp_state_dict, env_state_v, obs_v, _rng, env_params)
        runner_state, game_info = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["ppo"]["num_updates"]),
        )
        output = {"params": train_state.params, "batch_stats": train_state.batch_stats}
        return output
    
    return train


if __name__ == "__main__":
    config = parse_config()

    if config["ppo"]["use_wandb"]: 
        wandb.init(
            project = "LuxAIS3",
            name = config["ppo"]["run_name"],
            mode = "online",
        )
    jax.config.update("jax_numpy_dtype_promotion", "standard")

    ########### RUN PPO ###########
    train_jit = jax.jit(make_train(config, debug=True))
    rng = jax.random.PRNGKey(seed = config["ppo"]["seed"])
    output = train_jit(rng)

    if config["ppo"]["use_wandb"]: wandb.finish()
    
