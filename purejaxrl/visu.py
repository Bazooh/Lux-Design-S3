import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from luxai_s3.env import LuxAIS3Env
from luxai_s3.state import EnvState


def directional_arrow(action) -> tuple:
    """
    Return the directional arrow for the action.
    Args:
        action: The action.
    Returns:
        The directional arrow.
    """
    if action == 1:
        return 0, -1
    elif action == 2:
        return 1, 0
    elif action == 3:
        return 0, 1
    elif action == 4:
        return -1, 0
    else:
        return 0, 0


def visualize_grid(env_state: EnvState, logits_maps: jnp.ndarray) -> None:
    """
    Visualize the logits maps.
    Args:
        logits_maps: The logits maps. Shape: (24, 24, 6)
    """

    action_maps = jnp.argmax(logits_maps, axis=-1)

    fig, ax = plt.subplots()
    fig.set_size_inches(7, 7)

    # Color each relic in yellow
    for x, y in env_state.relic_nodes:
        ax.add_patch(Rectangle((x, y), 1, 1, color="yellow"))

    # Color each points in orange
    for i, relic_config in enumerate(env_state.relic_node_configs):
        for x in range(5):
            for y in range(5):
                if relic_config[x, y]:
                    relic_x, relic_y = env_state.relic_nodes[i]
                    ax.add_patch(
                        Rectangle(
                            (relic_x + x - 2, relic_y + y - 2),
                            1,
                            1,
                            color="orange",
                            fill=False,
                            linewidth=3,
                        )
                    )

    for x in range(24):
        for y in range(24):
            tile = env_state.map_features.tile_type[x, y]

            if tile == 1:
                ax.add_patch(Rectangle((x, y), 1, 1, color="purple", alpha=0.5))
            elif tile == 2:
                ax.add_patch(Rectangle((x, y), 1, 1, color="black", alpha=0.5))

    for i in range(24):
        for j in range(24):
            action = action_maps[i, j]
            if action == 0:
                continue

            if action == 5:
                ax.add_patch(Circle((i + 0.5, j + 0.5), 0.2, color="red"))
                continue

            x, y = i + 0.5, j + 0.5
            dx, dy = directional_arrow(action)
            new_i, new_j = round(i + dx), round(j + dy)
            if new_i >= 0 and new_i < 24 and new_j >= 0 and new_j < 24:
                new_action = action_maps[new_i, new_j]
                new_dx, new_dy = directional_arrow(new_action)
                if new_dx == -dx and new_dy == -dy:
                    x += 0.2 * dy
                    y += 0.2 * dx

            ax.arrow(
                x,
                y,
                0.6 * dx,
                0.6 * dy,
                head_width=0.3,
                head_length=0.3,
                fc="k",
                ec="k",
            )

    ax.set_xlim(-0.3, 24.3)
    ax.set_ylim(-0.3, 24.3)
    ax.set_xticks(jnp.arange(0, 25, 1))
    ax.set_yticks(jnp.arange(0, 25, 1))
    ax.grid(True)
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    import jax

    env = LuxAIS3Env()

    _, step_key = jax.random.split(jax.random.PRNGKey(0))
    _, env_state = env.reset(step_key)

    logits_maps = jax.random.uniform(jax.random.PRNGKey(0), (24, 24, 6))
    visualize_grid(env_state, logits_maps)
