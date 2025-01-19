import yaml
import jax
import flax

def parse_config(config_path = "purejaxrl/jax_config.yaml"):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    ###### Environment arguments ######    
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
    
    from purejaxrl.env.transform_reward import BasicPointReward, BasicExplorationReward,BasicEnergyReward, BasicFoundRelicReward
    if config_dict["env_args"]["transform_reward"] == "BasicPointReward":
        transform_reward = BasicPointReward()
    elif config_dict["env_args"]["transform_reward"] == "BasicExplorationReward":
        transform_reward = BasicExplorationReward()
    elif config_dict["env_args"]["transform_reward"] == "BasicEnergyReward":
        transform_reward = BasicEnergyReward()
    elif config_dict["env_args"]["transform_reward"] == "BasicFoundRelicReward":
        transform_reward = BasicFoundRelicReward()
    else:
        raise ValueError(f"Transform reward {config_dict['env_args']['transform_reward']} not supported")

    ###### Network arguments ######
    if config_dict["network"]["model"] == "HybridActorCritic":
        from purejaxrl.network import HybridActorCritic
        network = HybridActorCritic(transform_action.action_space.n)
    else:
        raise ValueError(f"Network {config_dict['network']['model']} not supported")
    
    if config_dict["network"]["checkpoint"] == "None":
        from purejaxrl.utils import init_network_params
        network_params = init_network_params(network=network, key=jax.random.PRNGKey(0), init_x=transform_obs.observation_space.sample(jax.random.PRNGKey(0)))
    else:
        raise ValueError(f"Network {config_dict['network']['name']} not supported")
        # from flax.training import orbax_utils
        # import orbax
        # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # # restored = orbax_checkpointer.restore(config_dict["network"]["checkpoint"])
        # # network_params = restored["model"]["params"]
    
    return {
        "network":{
            "model": network,
            "network_params": network_params,
        },
        "env_args":{
            "memory": memory,
            "transform_action": transform_action,
            "transform_obs": transform_obs,
            "transform_reward": transform_reward
        },
        "ppo": {
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
            "action_noise": float(config_dict["ppo"]["action_noise"]),
        }
    }