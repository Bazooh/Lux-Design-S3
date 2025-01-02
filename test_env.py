import numpy as np
from regex import B
from agents.lux.utils import Direction
from agents.models.dense import CNN
from agents.rl_agent import BasicRLAgent
from env_interface import EnvInterfaceForVec, EnvInterface
from gymnasium.vector import SyncVectorEnv


# vec_env = SyncVectorEnv([lambda: EnvInterfaceForVec() for _ in range(3)])

# vec_env.reset()

# print(vec_env.step([np.zeros((2, 16, 3), dtype=np.int32) for _ in range(3)]))

# env = EnvInterface()

# print(env.reset()[1])

env = EnvInterface()

_, env_params = env.reset()
player_0 = BasicRLAgent("player_0", env_params, "cpu", CNN(19))
player_1 = BasicRLAgent("player_1", env_params, "cpu", CNN(19))

player_0_actions = np.zeros((16, 3), dtype=np.int32)
player_1_actions = np.zeros((16, 3), dtype=np.int32)
_, _, _, _, _ = env.step(
    {
        "player_0": player_0_actions,
        "player_1": player_1_actions,
    }
)

player_0_actions[0, 0] = Direction.DOWN
player_1_actions[0, 0] = Direction.LEFT

obs, _, _, _, _ = env.step(
    {
        "player_0": player_0_actions,
        "player_1": player_1_actions,
    }
)

tensor_0 = player_0.obs_to_tensor(obs.player_0)[3]
tensor_1 = player_1.obs_to_tensor(obs.player_1)[3]

pass
