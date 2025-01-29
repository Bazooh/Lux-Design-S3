import jax.numpy as jnp
from functools import partial
import jax
from enum import IntEnum

class Direction(IntEnum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    
    
@jax.jit
def direction_to(src, target) -> Direction:
    delta = target - src 
    dx = delta[0]
    dy = delta[1]   
    direction = jax.lax.select(
        (delta == 0).all(),
        Direction.CENTER,
        jax.lax.select(
            jnp.abs(dx) > jnp.abs(dy),
            jax.lax.select(
                dx > 0,
                Direction.RIGHT,
                Direction.LEFT
            ),
            jax.lax.select(
                dy > 0,
                Direction.DOWN,
                Direction.UP
            )
        )
    )
    return direction

def create_manhattan_matrix(n):
    center = (n // 2, n // 2)
    x = jnp.arange(n)
    y = jnp.arange(n)
    xv, yv = jnp.meshgrid(x, y, indexing="ij")
    D = jnp.abs(xv - center[0]) + jnp.abs(yv - center[1])
    return D

@partial(jax.jit, static_argnums=(1))
def find_nearest(mask_image, n):
    D = create_manhattan_matrix(n)
    idx = jnp.argmin(D * mask_image + 100 * (1 - mask_image))
    return direction_to(src = jnp.ones(2) * n//2, target =  jnp.array([idx // n, idx % n]))

if __name__ == "__main__":
    relic = jnp.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    relic = jax.scipy.signal.convolve2d((relic  == 1).astype(jnp.int8), jnp.ones((5, 5)), mode='same') >0
    print(relic)
    pos = jnp.array([2, 2])
    relic_mask_image = jax.lax.dynamic_slice((relic == 1), (1, 1), (5, 5))
    print(relic_mask_image)
    print(find_nearest(relic_mask_image, 5))