import yaml
import jax
import flax

def parse_config(config_path = "purejaxrl/jax_config.yaml"):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    ###### Environment arguments ######    
    num_stats = len(config_dict["env_args"]["reward_weights"])//2
    reward_weights = {
        config_dict["env_args"]["reward_weights"]["stat_" + str(i)]: config_dict["env_args"]["reward_weights"]["weight_" + str(i)] for i in range(num_stats)
    }

    if config_dict["env_args"]["memory"] == "RelicPointMemory":
        from purejaxrl.env.memory import RelicPointMemory
        memory = RelicPointMemory()
    else:
        raise ValueError(f"Memory {config_dict['env_args']['memory']} not supported")
    
    if config_dict["env_args"]["transform_action"] == "SimplerActionNoSap":
        from purejaxrl.env.transform_action import SimplerActionNoSap
        transform_action = SimplerActionNoSap()
    else:
        raise ValueError(f"Transform action {config_dict['env_args']['transform_action']} not supported")
    
    if config_dict["env_args"]["transform_obs"] == "HybridTransformObs":
        from purejaxrl.env.transform_obs import HybridTransformObs
        transform_obs = HybridTransformObs()
    else:
        raise ValueError(f"Transform obs {config_dict['env_args']['transform_obs']} not supported")


    ###### Network arguments ######
    if config_dict["network"]["model"] == "Pix2Pix_AC":
        from purejaxrl.network import Pix2Pix_AC
        model = Pix2Pix_AC(transform_action.action_space.n)
    else:
        raise ValueError(f"Network {config_dict['network']['model']} not supported")
    
    if config_dict["network"]["load_from_checkpoint"] == "None":
        from purejaxrl.utils import init_state_dict
        state_dict = init_state_dict(model=model, key=jax.random.PRNGKey(0), init_x=transform_obs.observation_space.sample(jax.random.PRNGKey(0)), print_summary = False)
    else:
        raise ValueError(f"Network {config_dict['network']['name']} not supported")
        # from flax.training import orbax_utils
        # import orbax
        # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # # restored = orbax_checkpointer.restore(config_dict["network"]["checkpoint"])
        # # network_params = restored["model"]["params"]
    
    ######## Arena Agent  #########
    if config_dict["ppo"]["arena_agent"] == "NaiveAgent_Jax":
        from rule_based_jax.naive.agent import NaiveAgent_Jax
        arena_agent = NaiveAgent_Jax(player = "player_1")
    elif config_dict["ppo"]["arena_agent"] == "RandomAgent_Jax":
        from rule_based_jax.random.agent import RandomAgent_Jax
        arena_agent = RandomAgent_Jax(player =  "player_1")

    return {
        "network":{
            "model": model,
            "state_dict": state_dict,
        },
        "env_args":{
            "reward_weights": reward_weights, 
            "memory": memory,
            "transform_action": transform_action,
            "transform_obs": transform_obs,
        },
        "ppo": {
            # Learning args
            "match_count_per_episode": int(config_dict["ppo"]["match_count_per_episode"]),
            "lr": float(config_dict["ppo"]["lr"]),
            "num_envs": int(config_dict["ppo"]["num_envs"]),
            "num_steps": int(config_dict["ppo"]["num_steps"]),
            "total_timesteps": float(config_dict["ppo"]["total_timesteps"]),
            "update_epochs": int(config_dict["ppo"]["update_epochs"]),
            "num_minibatches": int(config_dict["ppo"]["num_minibatches"]),
            "gamma": float(config_dict["ppo"]["gamma"]),
            "gae_lambda": float(config_dict["ppo"]["gae_lambda"]),
            "clip_grad_norm": float(config_dict["ppo"]["clip_grad_norm"]),
            "clip_eps": float(config_dict["ppo"]["clip_eps"]),
            "ent_coef": float(config_dict["ppo"]["ent_coef"]),
            "vf_coef": float(config_dict["ppo"]["vf_coef"]),
            "max_grad_norm": float(config_dict["ppo"]["max_grad_norm"]),
            "anneal_lr": bool(config_dict["ppo"]["anneal_lr"]),
            "seed": int(config_dict["ppo"]["seed"]),
            "action_temperature": float(config_dict["ppo"]["action_temperature"]),
            # Log args
            "use_wandb": bool(config_dict["ppo"]["use_wandb"]),
            "arena_agent": arena_agent,
            "record_freq": int(config_dict["ppo"]["record_freq"]),
            "arena_freq": int(config_dict["ppo"]["arena_freq"]),
            "match_count_per_episode_arena": int(config_dict["ppo"]["match_count_per_episode_arena"]),
        }
    }