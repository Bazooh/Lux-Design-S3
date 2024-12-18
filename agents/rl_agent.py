import numpy as np
import torch
from luxai_s3.state import EnvObs
from agents.models.dense import CNN
from agents.tensor_converters.tensor import TensorConverter, BasicMapExtractor
from agents.reward_shapers.reward import RewardShaper, DefaultRewardShaper, MixingRewardShaper 
from agents.base_agent import Agent, N_Actions, N_Agents


class RLAgent(Agent):
    def __init__(
        self,
        player: str,
        env_cfg: dict[str, int],
        model: torch.nn.Module,
        tensor_converter: TensorConverter,  
        reward_shaper: RewardShaper,
    ) -> None:
        super().__init__(player, env_cfg)
        self.model = model
        self.tensor_converter = tensor_converter
        self.reward_shaper = reward_shaper


    def actions(
        self, obs: EnvObs, remainingOverageTime: int = 60
    ) -> np.ndarray[tuple[N_Agents, N_Actions], np.dtype[np.int32]]:
        return (
            self.sample_action(self.obs_to_tensor(obs), epsilon=0)
            .squeeze(0)
            .detach()
            .numpy()
        )
    
    def sample_action(self, obs_tensor: torch.Tensor, epsilon: float):
        raise NotImplementedError

    def sample_action(self, obs_tensor: torch.Tensor, epsilon: float):
        """Returns a tensor of shape (batch_size, n_agents, 3) with the actions to take"""
        raise NotImplementedError

    def obs_to_tensor(self, obs: EnvObs) -> torch.Tensor:
        return self.tensor_converter.convert(obs, self.team_id)

    def save_net(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
    
    def load_net(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))


class BasicRLAgent(RLAgent):
    def __init__(
        self,
        player: str,
        env_cfg: dict[str, int],
        model: CNN | None = None,
        model_path: str | None = None,
    ) -> None:
        assert (
            model is None or model_path is None
        ), "Only one of model or model_path can be provided"

        if model_path is not None:
            model = CNN()
            model.load_state_dict(torch.load(model_path))

        super().__init__(
            player = player, 
            env_cfg=env_cfg, 
            model = model if model is not None else CNN(), 
            tensor_converter = BasicMapExtractor(),
            reward_shaper = DefaultRewardShaper(),
        )


    def sample_action(self, obs_tensor: torch.Tensor, epsilon: float):
        """Returns a tensor of shape (batch_size, n_agents, 3) with the actions to take"""
        # ! WARNING ! : This function does not use the sap action (it only moves the units)
        out = self.model(obs_tensor)

        batch_size, n_agents, n_actions = out.shape

        mask = torch.rand((batch_size,)) <= epsilon
        action = torch.zeros((batch_size, n_agents, 3), dtype=torch.int32)
        action[mask, :, 0] = torch.randint(0, n_actions, action[mask].shape[:2]).int()
        action[~mask, :, 0] = out[~mask].argmax(dim=2).int()
        return action