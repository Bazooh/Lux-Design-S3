import yaml
import jax

def parse_config(config_path = "purejaxrl/jax_config.yaml"):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    ###### Environment arguments ######

    if config_dict["env_args"]["symmetry"] == "ActionAndObsSymmetry":
        from purejaxrl.wrappers.symmetry import ActionAndObsSymmetry
        symmetry = ActionAndObsSymmetry()
    else:
        raise ValueError(f"Symmetry {config_dict['env_args']['symmetry']} not supported")
    
    if config_dict["env_args"]["memory"] == "RelicPointMemory":
        from purejaxrl.wrappers.memory import RelicPointMemory
        memory = RelicPointMemory()
    else:
        raise ValueError(f"Memory {config_dict['env_args']['memory']} not supported")
    
    if config_dict["env_args"]["transform_action"] == "SimplerActionNoSap":
        from purejaxrl.wrappers.transform_action import SimplerActionNoSap
        transform_action = SimplerActionNoSap()
    else:
        raise ValueError(f"Transform action {config_dict['env_args']['transform_action']} not supported")
    
    if config_dict["env_args"]["transform_obs"] == "HybridTransformObs":
        from purejaxrl.wrappers.transform_obs import HybridTransformObs
        transform_obs = HybridTransformObs()
    else:
        raise ValueError(f"Transform obs {config_dict['env_args']['transform_obs']} not supported")
    
    if config_dict["env_args"]["transform_reward"] == "BasicPointBasedReward":
        from purejaxrl.wrappers.transform_reward import BasicPointBasedReward
        transform_reward = BasicPointBasedReward()
    else:
        raise ValueError(f"Transform reward {config_dict['env_args']['transform_reward']} not supported")

    ###### Network arguments ######
    if config_dict["network"]["name"] == "HybridActorCritic":
        from purejaxrl.network import HybridActorCritic
        network = HybridActorCritic(transform_action.action_space.n)
    else:
        raise ValueError(f"Network {config_dict['network']['name']} not supported")
    
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
            "network": network,
            "network_params": network_params,
        },
        "env_args":{
            "symmetry": symmetry,
            "memory": memory,
            "transform_action": transform_action,
            "transform_obs": transform_obs,
            "transform_reward": transform_reward
        }
    }