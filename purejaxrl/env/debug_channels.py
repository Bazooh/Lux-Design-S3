import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import jax, chex
from typing import Any
from purejaxrl.utils import sample_params
from luxai_s3.env import LuxAIS3Env, EnvObs, EnvParams
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from purejaxrl.jax_agent import JaxAgent, RawJaxAgent

def EnvObs_to_dict(obs: EnvObs) ->  dict[str, Any]:
    return {
        "units": {
            "position": obs.units.position,
            "energy": obs.units.energy,
        },
        "units_mask": obs.units_mask,
        "sensor_mask": obs.sensor_mask,
        "map_features": {
            "energy": obs.map_features.energy,
            "tile_type": obs.map_features.tile_type,
        },
        "relic_nodes": obs.relic_nodes,
        "relic_nodes_mask": obs.relic_nodes_mask,
        "team_points": obs.team_points,
        "team_wins": obs.team_wins,
        "steps": obs.steps,
        "match_steps": obs.match_steps
    }

def rollout(
        agent_0: JaxAgent, 
        agent_1: JaxAgent, 
        actor_0: Any,
        actor_1: Any,
        vanilla_env: LuxAIS3Env, 
        env_params,
        key: chex.PRNGKey,
        steps = 50
    ):
    """
    Evaluate the trained agent against a reference agent using a separate eval environment.
    """
    rng, _rng = jax.random.split(key)
    obs, env_state = vanilla_env.reset(rng, env_params)

    stack_obs_0 = [] # Contains the observations of the agent 0
    stack_obs_1 = [] # Contains the observations of the agent 1
    stack_relic_w = []
    for step_idx in tqdm(range(steps)):
        # redo the observation conversion: agent_0 and agent_1 
        agent_0.memory_state = agent_0.memory.update(obs = obs["player_0"], team_id=agent_0.team_id, memory_state=agent_0.memory_state)
        agent_1.memory_state = agent_1.memory.update(obs = obs["player_1"], team_id=agent_1.team_id, memory_state=agent_1.memory_state)
        expanded_obs_0 = agent_0.memory.expand(obs["player_0"], agent_0.team_id, agent_0.memory_state)
        expanded_obs_1 = agent_1.memory.expand(obs["player_1"], agent_1.team_id, agent_1.memory_state)
        transformed_obs_0 = agent_0.transform_obs.convert(team_id=agent_0.team_id, obs = expanded_obs_0, params=env_params, memory_state=agent_0.memory_state)
        transformed_obs_1 = agent_1.transform_obs.convert(team_id=agent_1.team_id, obs = expanded_obs_1, params=env_params, memory_state=agent_1.memory_state)
        stack_obs_0.append(transformed_obs_0)
        stack_obs_1.append(transformed_obs_1)
        stack_relic_w.append(env_state.relic_nodes_map_weights)
        action = {
            "player_0": actor_0.act(step = step_idx, obs = EnvObs_to_dict(obs["player_0"])), 
            "player_1": actor_1.act(step = step_idx, obs = EnvObs_to_dict(obs["player_1"])),
        }
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, truncated_dict, terminated_dict, info = vanilla_env.step(rng, env_state, action, env_params)

    channel_names = list(agent_0.transform_obs.image_features.keys())
    channels_p0 = np.stack([obs["image"] for obs in stack_obs_0])
    channels_p1 = np.stack([obs["image"] for obs in stack_obs_1])
    vector_names = list(agent_0.transform_obs.vector_features.keys())
    vector_p0 = np.stack([obs["vector"] for obs in stack_obs_0])
    vector_p1 = np.stack([obs["vector"] for obs in stack_obs_1])
    print(channels_p0.shape, channels_p1.shape, len(channel_names))
    print(vector_p0.shape, vector_p1.shape, len(vector_names))
    stack_relic_w = np.stack(stack_relic_w)
    return channels_p0, channels_p1, channel_names, vector_p0, vector_p1, vector_names, stack_relic_w

COLUMNS = 15

# Function to plot tensor features
def plot_channel_features(channels: np.ndarray, axes_row: np.ndarray, title_prefix: str, channel_names: list[str]):
    """Plots the tensor features across one row of subplots."""
    for i in range(len(channel_names)):
        ax = axes_row[i]
        ax.clear()
        ax.imshow(channels[i].T)
        ax.set_title(f"{title_prefix} {i}: {channel_names[i]}", fontsize=6)
        ax.axis("off")
    for i in range(len(channel_names), COLUMNS):
        ax = axes_row[i]
        ax.clear()
        ax.axis("off")
# Function to plot vector features
def plot_vector_features(vectors: np.ndarray, axes_row: np.ndarray, title_prefix: str, vector_names: list[str]):
    """
    Plots the vector features as horizontal heatmaps using the same colormap and overlays scalar values.
    vectors: np.ndarray of shape (N_features,)
    axes_row: Axes for the corresponding row
    """
    for i in range(len(vector_names)):
        ax = axes_row[i]
        ax.clear()
        # Create a horizontal heatmap-like bar for the vector values
        heatmap = np.tile(vectors[i], (10, 10))  # Repeat values for visualization
        ax.imshow(heatmap, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto", origin="upper")
        ax.set_title(f"{title_prefix} {i}: {vector_names[i]}", fontsize=6)
        ax.axis("off")
        
        # Overlay the scalar value in large text
        scalar_value = f"{vectors[i]:.2f}"  # Format the scalar value
        ax.text(
            heatmap.shape[1] // 2,  # Center horizontally
            heatmap.shape[0] // 2,  # Center vertically
            scalar_value,
            color="black",  # Adjust text color for contrast
            fontsize=12,
            ha="center",
            va="center",
            fontweight="bold"
        )

# Function to plot relic weights
def plot_relic_weights(relic_weights: np.ndarray, axes_row: np.ndarray, title_prefix: str):
    """Plots the relic weights as a heatmap in the fifth row."""
    for i, ax in enumerate(axes_row):
        ax.clear()
        ax.axis("off")
        if i == 0:  # Display the heatmap only in the first column
            ax.imshow(relic_weights.T, cmap="viridis", origin="upper")
            ax.set_title(f"{title_prefix} Relic Weights", fontsize=6)


fig, axes = plt.subplots(5, COLUMNS, figsize=(25, 8))

# Update function for animation
def update(frame_idx: int, 
           channels: list[list[np.ndarray]], vectors: list[list[np.ndarray]], relic_weights, 
           channel_names: list[str], vector_names: list[str], progressbar):
    """Updates the plot for a given frame."""
    fig.suptitle(f"Frame {frame_idx}", fontsize=16)
    # Player 0 Channels (Row 0)
    plot_channel_features(channels[0][frame_idx], axes[0, :], "P0-Chan.", channel_names)
    # Player 0 Vectors (Row 1)
    plot_vector_features(vectors[0][frame_idx], axes[1, :len(vector_names)], "P0-Vec.", vector_names)
    # Player 1 Channels (Row 2)
    plot_channel_features(channels[1][frame_idx], axes[2, :], "P1-Chan.", channel_names)
    # Player 1 Vectors (Row 3)
    plot_vector_features(vectors[1][frame_idx], axes[3, :len(vector_names)], "P1-Vec.", vector_names)
    plot_relic_weights(relic_weights[frame_idx], axes[4, :], "Relic Weights")
    progressbar.update(1)




if __name__ == "__main__":
    from purejaxrl.jax_agent import JaxAgent, RawJaxAgent
    from rule_based.random.agent import RandomAgent
    from rule_based.relicbound.agent import RelicboundAgent
    from rule_based.naive.agent import NaiveAgent
    from tqdm import tqdm
    # RUN MATCH
    seed = np.random.randint(0, 100)
    key = jax.random.PRNGKey(seed)
    vanilla_env = LuxAIS3Env(auto_reset=True)
    env_params = sample_params(key)
    
    channels_p0, channels_p1, channel_names, vector_p0, vector_p1, vector_names, relic_w = rollout(
        agent_0 = JaxAgent("player_0", env_params.__dict__),
        agent_1 = JaxAgent("player_1", env_params.__dict__),
        actor_0 = NaiveAgent("player_0", env_params.__dict__),
        actor_1 = NaiveAgent("player_1", env_params.__dict__),
        key = key,
        vanilla_env = vanilla_env,
        env_params = env_params,
        steps=100
    )
    
    channels = np.stack([channels_p0, channels_p1])
    vector = np.stack([vector_p0, vector_p1])
    n_frames = len(channels_p0)
    
    progressbar = tqdm(total=n_frames)

    anim = FuncAnimation(
        fig,
        lambda frame_idx: update(
            frame_idx, 
            channels=channels, vectors=vector, relic_weights=relic_w, 
            channel_names=channel_names, vector_names=vector_names, progressbar=progressbar
        ),
        frames=n_frames,
        interval=250,
    )

    anim.save("purejarxrl_debug_channels.gif")
    plt.close(fig)
    print("GIF saved as 'purejarxrl_debug_channels.gif'")
