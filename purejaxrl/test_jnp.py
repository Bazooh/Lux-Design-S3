import jax
import jax.numpy as jnp
from env.utils import mirror_position

    
def symmetrize_relic_positions_arrays(relic_positions):
    """"""
    relic_pos = relic_positions.reshape((2, 3, 2))
    mirrored_relic_pos = jax.vmap(jax.vmap(mirror_position))(relic_pos)
    empty = -jnp.ones((3,2))
    relic_positions = jnp.stack([
        jnp.where(relic_pos[0] > 0, relic_pos[0], jnp.where(relic_pos[1] > 0, mirrored_relic_pos[1], empty)), 
        jnp.where(relic_pos[1] > 0, relic_pos[1], jnp.where(relic_pos[0] > 0, mirrored_relic_pos[0], empty))
    ]
    )
    return relic_positions.reshape((6, 2))

if __name__ == "__main__":
    relic_positions = jnp.array([[2,3],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[6,6]])
    sym = symmetrize_relic_positions_arrays(relic_positions)
    print(sym)