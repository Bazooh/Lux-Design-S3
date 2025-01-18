import numpy as np
from env_interface import EnvInterfaceForVec
from gymnasium.vector import SyncVectorEnv


# vec_env = SyncVectorEnv([lambda: EnvInterfaceForVec() for _ in range(3)])

# obs, info = vec_env.reset()
# obs, a, b, c, d = vec_env.step(np.zeros((3, 2, 16, 3), dtype=np.int32))
# print(a)
# print(b)
# print(type(c))
# print(d.keys())

import torch

# Create a tensor of shape (6, 3) where 2*n=6 and p=3
input_tensor = torch.tensor([
    [1, 2, 3],  # Row 0
    [4, 5, 6],  # Row 1
    [7, 8, 9],  # Row 2
    [10, 11, 12],  # Row 3
    [13, 14, 15],  # Row 4
    [16, 17, 18],  # Row 5
])

# Perform the transformation
n = input_tensor.shape[0] // 2
result = input_tensor.view(2, n, -1).permute(1, 0, 2)

print("Input Tensor:")
print(input_tensor)
print("\nResult Tensor:")
print(result)
