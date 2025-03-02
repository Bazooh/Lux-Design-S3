from dataclasses import dataclass
from pytorch.env.memory import RelicPointMemory
from pytorch.env.transform_action import SimplerActionWithSap
from pytorch.env.transform_obs import HybridTransformObs
from typing import Any
@dataclass
class Args:
    # Env Args
    memory: Any = RelicPointMemory()
    transform_action: Any = SimplerActionWithSap()
    transform_obs: Any = HybridTransformObs()
    cuda: bool = True
    # network args
    n_resblocks: int = 5
    n_channels: int = 128
    embedding_time: int = 8
    normalize_logits: bool = True
    normalize_value: bool = True
    action_masking: bool = False
    # Algorithm specific arguments
    track: bool = False
    seed: int = 0
    torch_deterministic: bool = True
    total_timesteps: int = 1e8
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.95
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

