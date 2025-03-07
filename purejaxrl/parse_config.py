import yaml
import jax
import orbax, os
from datetime import datetime
from ROOT_DIR import ROOT_DIR
def parse_config(yaml_path = "purejaxrl/jax_config.yaml"):
    config = {}

    with open(yaml_path, 'r') as file:
        yaml_dict = yaml.safe_load(file)

    ###### Environment arguments ######    
    if yaml_dict["env_args"]["memory"] == "RelicPointMemory":
        from purejaxrl.env.memory import RelicPointMemory
        memory = RelicPointMemory()
    else:
        raise ValueError(f"Memory {yaml_dict['env_args']['memory']} not supported")
    
    if yaml_dict["env_args"]["transform_action"] == "SimplerActionNoSap":
        from purejaxrl.env.transform_action import SimplerActionNoSap
        transform_action = SimplerActionNoSap()
    elif yaml_dict["env_args"]["transform_action"] == "SimplerActionWithSap":
        from purejaxrl.env.transform_action import SimplerActionWithSap
        transform_action = SimplerActionWithSap()
    else:
        raise ValueError(f"Transform action {yaml_dict['env_args']['transform_action']} not supported")
    
    if yaml_dict["env_args"]["transform_obs"] == "HybridTransformObs":
        from purejaxrl.env.transform_obs import HybridTransformObs
        transform_obs = HybridTransformObs()
    else:
        raise ValueError(f"Transform obs {yaml_dict['env_args']['transform_obs']} not supported")

    config["env_args"] = {
        "transform_obs": transform_obs,
        "transform_action": transform_action,
        "memory": memory,
        "reward_smoothing": bool(yaml_dict["env_args"]["reward_smoothing"]),
    }

    ######### Reward arguments ########
    if "reward_phases" in yaml_dict["env_args"].keys():
        from purejaxrl.env.wrappers import RewardObject, RewardType
        reward_phases = []
        
        for i, reward_phase in enumerate(yaml_dict["env_args"]['reward_phases']):
            reward_phase = yaml_dict["env_args"]['reward_phases'][i]
            reward_type = RewardType(reward_phase["type"])
            reward_weights = {k: float(v) for k, v in reward_phase["weights"].items()}
            reward_phases.append(RewardObject(reward_type, reward_weights))
            
        config["env_args"]["reward_phases"] = reward_phases


    ###### Network arguments ######
    network_args = {
            "action_dim": int(transform_action.action_space.n),
            "n_resblocks": int(yaml_dict["network"]["n_resblocks"]),
            "n_channels": int(yaml_dict["network"]["n_channels"]),
            "embedding_time": int(yaml_dict["network"]["embedding_time"]),
            "normalize_logits": bool(yaml_dict["network"]["normalize_logits"]),
            "normalize_value": bool(yaml_dict["network"]["normalize_value"]),
            "action_masking": bool(yaml_dict["network"]["action_masking"])
    }
    if yaml_dict["network"]["model"] == "Pix2Pix_AC":
        from purejaxrl.network import Pix2Pix_AC
        model = Pix2Pix_AC(**network_args)
    else:
        raise ValueError(f"Network {yaml_dict['network']['model']} not supported")
    
    
    from purejaxrl.utils import init_state_dict
    init_state = init_state_dict(model=model, key=jax.random.PRNGKey(0), init_x=transform_obs.observation_space.sample(jax.random.PRNGKey(0)), print_summary = False)

    if yaml_dict["network"]["load_from_checkpoint"] == "None":
        state_dict = init_state
    else:
        from purejaxrl.utils import restore_state_dict, restore_state_dict_cpu
        checkpoint_path = ROOT_DIR + yaml_dict["network"]["load_from_checkpoint"]
        if not os.path.exists(checkpoint_path): raise ValueError(f"Checkpoint {checkpoint_path} not found") 
        if yaml_dict["network"]["restore_to_cpu"]: state_dict = restore_state_dict_cpu(checkpoint_path)
        else: state_dict = restore_state_dict(checkpoint_path)
    config["network"]={
            "model": model,
            "state_dict": state_dict,
    }
    config["network_args"] = network_args

    ######## PPO arguments  #########
    if "ppo" in yaml_dict.keys():
        config["ppo"] = {
            # Learning args
            "match_count_per_episode": int(yaml_dict["ppo"]["match_count_per_episode"]),
            "start_lr": float(yaml_dict["ppo"]["start_lr"]),
            "end_lr": float(yaml_dict["ppo"]["end_lr"]),
            "num_envs": int(yaml_dict["ppo"]["num_envs"]),
            "num_steps": int(yaml_dict["ppo"]["num_steps"]),
            "total_timesteps": float(yaml_dict["ppo"]["total_timesteps"]),
            "update_epochs": int(yaml_dict["ppo"]["update_epochs"]),
            "num_minibatches": int(yaml_dict["ppo"]["num_minibatches"]),
            "gamma": float(yaml_dict["ppo"]["gamma"]),
            "gamma_smoothing": bool(yaml_dict["ppo"]["gamma_smoothing"]),
            "gae_lambda": float(yaml_dict["ppo"]["gae_lambda"]),
            "clip_grad_norm": float(yaml_dict["ppo"]["clip_grad_norm"]),
            "clip_eps": float(yaml_dict["ppo"]["clip_eps"]),
            "points_pred_coef": float(yaml_dict["ppo"]["points_pred_coef"]),
            "ent_coef": float(yaml_dict["ppo"]["ent_coef"]),
            "vf_coef": float(yaml_dict["ppo"]["vf_coef"]),
            "max_grad_norm": float(yaml_dict["ppo"]["max_grad_norm"]),
            "anneal_lr": bool(yaml_dict["ppo"]["anneal_lr"]),
            "seed": int(yaml_dict["ppo"]["seed"]),
            "selfplay_freq_update": int(yaml_dict["ppo"]["selfplay_freq_update"]),
            "selfplay_window": int(yaml_dict["ppo"]["selfplay_window"]),
            "play_against_latest_model_ratio": float(yaml_dict["ppo"]["play_against_latest_model_ratio"]),
            # Log args
            "verbose": int(yaml_dict["ppo"]["verbose"]),
            "use_wandb": bool(yaml_dict["ppo"]["use_wandb"]),
            "run_name": yaml_dict["ppo"]["run_name"]+"_"+datetime.now().strftime("%Y_%m_%d"),
            # Save Args
            "save_checkpoint_path": ROOT_DIR + "/checkpoints/" + yaml_dict["ppo"]["save_checkpoint_path"]+"_"+datetime.now().strftime("%Y_%m_%d"),
            "save_checkpoint_freq": int(yaml_dict["ppo"]["save_checkpoint_freq"]),
        }

    ######## Arena Against Jax Agent  #########
    if "arena_jax" in yaml_dict.keys():
        if yaml_dict["arena_jax"]["agent"] == "RandomAgent_Jax":
            from rule_based_jax.random.agent import RandomAgent_Jax
            arena_agent = RandomAgent_Jax
        elif yaml_dict["arena_jax"]["agent"] == "NaiveAgent_Jax":
            from rule_based_jax.naive.agent import NaiveAgent_Jax
            arena_agent = NaiveAgent_Jax
        else:
            raise ValueError(f"Agent {yaml_dict['arena_jax']['agent']} not supported")
        config["arena_jax"] = {
            "agent": arena_agent,
            "num_matches": int(yaml_dict["arena_jax"]["num_matches"]),
            "arena_freq": int(yaml_dict["arena_jax"]["arena_freq"]),
            "record_freq": int(yaml_dict["arena_jax"]["record_freq"]),
        }
    else:
        config["arena_jax"] = None

    ######## Arena Against Standard Agent  #########
    if "arena_std" in yaml_dict.keys():
        if yaml_dict["arena_std"]["agent"] == "RandomAgent":
            from rule_based.random.agent import RandomAgent
            arena_agent = RandomAgent
        elif yaml_dict["arena_std"]["agent"] == "NaiveAgent":
            from rule_based.naive.agent import NaiveAgent
            arena_agent = NaiveAgent
        elif yaml_dict["arena_std"]["agent"] == "RelicBoundAgent":
            from rule_based.relicbound.agent import RelicboundAgent
            arena_agent = RelicboundAgent
        else:
            raise ValueError(f"Agent {yaml_dict['arena_std']['agent']} not supported")
        config["arena_std"] = {
            "agent": arena_agent,
            "num_matches": int(yaml_dict["arena_std"]["num_matches"]),
            "arena_freq": int(yaml_dict["arena_std"]["arena_freq"]),
            "record_freq": int(yaml_dict["arena_std"]["record_freq"]),
        }
    else:
        config["arena_std"] = None
    
    return config