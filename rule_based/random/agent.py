from luxai_s3.state import EnvObs
import numpy as np
class RandomAgent:
    def __init__(self, player: str, env_cfg) -> None:
        pass
    def act(
        self, step: int, obs: EnvObs, remainingOverageTime: int = 60
    ):
        actions = np.zeros((16, 3), dtype=np.int32)
        actions[:,0] = np.random.randint(0, 5, (16,), dtype=np.int32)
        return actions
