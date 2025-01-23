import jax.numpy as jnp
import jax

array = jnp.arange(16).reshape(4, 4) 
from utils import symmetrize, mirror_grid
print(array)
print(symmetrize(team_id = 0, array = array))
print(symmetrize(team_id = 1, array = array))