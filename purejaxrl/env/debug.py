from math import e
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import jax, chex
from typing import Any
from purejaxrl.utils import sample_params
from luxai_s3.env import EnvObs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from purejaxrl.jax_agent import JaxAgent, RawJaxAgent
from tqdm import tqdm
from purejaxrl.env.make_env import make_vanilla_env, TrackerWrapper, LogWrapper
from purejaxrl.parse_config import parse_config
from rule_based.naive.agent import NaiveAgent

def EnvObs_to_dict(obs: EnvObs) -> dict[str, Any]:
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
        vanilla_env: TrackerWrapper, 
        env_params,
        key: chex.PRNGKey,
        steps = 50
    ):
    rng, _rng = jax.random.split(key)
    obs, env_state = vanilla_env.reset(rng, env_params)

    stack_images = []
    stack_stats = []
    point_weights = env_state.relic_nodes_map_weights
    for step_idx in tqdm(range(steps), desc = "Rollout and store obs+stats"):
        agent_0.memory_state = agent_0.memory.update(obs=obs["player_0"], team_id=agent_0.team_id, memory_state=agent_0.memory_state)
        agent_1.memory_state = agent_1.memory.update(obs=obs["player_1"], team_id=agent_1.team_id, memory_state=agent_1.memory_state)

        transformed_obs_0 = agent_0.transform_obs.convert(team_id=agent_0.team_id, obs=obs["player_0"], params=env_params, memory_state=agent_0.memory_state)
        transformed_obs_1 = agent_1.transform_obs.convert(team_id=agent_1.team_id, obs=obs["player_1"], params=env_params, memory_state=agent_1.memory_state)

        action = {
            "player_0": actor_0.act(step=step_idx, obs=EnvObs_to_dict(obs["player_0"])), 
            "player_1": actor_1.act(step=step_idx, obs=EnvObs_to_dict(obs["player_1"])),
        }
        rng, _rng = jax.random.split(rng)
        obs, env_state, _, _, info = vanilla_env.step(rng, env_state, action, env_params)

        stack_images.append((transformed_obs_0["image"], transformed_obs_1["image"]))
        stack_stats.append((info["episode_stats_player_0"], info["episode_stats_player_1"]))

    channels_arrays = {
        "obs_player_0": {feat: np.array([stack_images[i][0][feat_idx] for i in range(len(stack_images))], dtype=np.float32) for feat_idx,feat in enumerate(agent_0.transform_obs.image_features)},
        "obs_player_1": {feat: np.array([stack_images[i][1][feat_idx] for i in range(len(stack_images))], dtype=np.float32) for feat_idx,feat in enumerate(agent_1.transform_obs.image_features)}
    }
    stats_arrays = {
        "episode_stats_player_0": {stat: np.array([getattr(stack_stats[i][0], stat) for i in range(len(stack_stats))]) for stat in vanilla_env.stats_names},
        "episode_stats_player_1": {stat: np.array([getattr(stack_stats[i][1], stat) for i in range(len(stack_stats))]) for stat in vanilla_env.stats_names}
    }
    vanilla_env.close()

    return channels_arrays, stats_arrays, point_weights


# Function to plot tensor features
def plot_channel_features(channels: dict, axes_row: np.ndarray, title_prefix: str, frame_idx: int, n_columns: int, relic_weights):
    for i, (feat_name, feat_array) in enumerate(channels.items()):
        ax = axes_row[i]
        ax.clear()
        if feat_name in ["Ally_Units", "Enemy_Units"]: 
            ax.imshow(feat_array[frame_idx].T, cmap = "bwr", vmin = -2, vmax = 2, aspect = "auto")
        if feat_name in ["Points", "Relic"]: 
            ax.imshow(feat_array[frame_idx].T, cmap = "bwr", vmin = -1, vmax = 1, aspect = "auto")
        else:
            ax.imshow(feat_array[frame_idx].T, cmap = "bwr", aspect = "auto")
        ax.set_title(f"{title_prefix} {feat_name}", fontsize=10)
        ax.grid(True)
        ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
        

    # Plot relic weights
    ax = axes_row[i+1]
    ax.clear()
    ax.imshow(relic_weights.T, cmap = "bwr", vmin = -1, vmax = 1, aspect = "auto")
    ax.set_title(f"(Ground Truth) Points", fontsize=10)
    ax.grid(True)
    ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    

    for i in range(len(channels.items())+1, n_columns):
        ax = axes_row[i]
        ax.clear()
        ax.xticks("off")

# Function to plot stat features
def plot_stat_features(stats_p0: dict, stats_p1: dict, axes_row: np.ndarray, title_prefix: str, frame_idx: int, n_columns: int):
    for i,stat_name in enumerate(stats_p0.keys()):
        ax = axes_row[i]
        ax.clear()

        stat_p0 = stats_p0[stat_name][:frame_idx + 1]
        stat_p1 = stats_p1[stat_name][:frame_idx + 1]

        ax.plot(stat_p0[:frame_idx + 1], color="blue", linewidth=3, alpha=0.75)
        ax.plot(stat_p1[:frame_idx + 1], color="red", linewidth=3, alpha=0.75)
        ax.set_title(f"{title_prefix} {stat_name}", fontsize=10)
        ax.set_xlim(0, len(stats_p0[stat_name]))

        y_min = min(stats_p0[stat_name].min(), stats_p1[stat_name].min())
        y_max = max(stats_p0[stat_name].max(), stats_p1[stat_name].max())
        if y_min != y_max: ax.set_ylim(y_min, y_max)

    for i in range(len(stats_p0.keys()), n_columns):
        ax = axes_row[i]
        ax.clear()
        ax.axis("off")


# Update function for animation
def update(frame_idx: int, 
            channels: dict, 
            stats: dict,
            point_weights,
            n_columns, 
            progressbar):
    
    fig.suptitle(f"Frame {frame_idx}", fontsize=16)
    # Player 0 Channels (Row 0)
    plot_channel_features(channels["obs_player_0"], axes[0, :], "P0-", frame_idx, n_columns, relic_weights=point_weights)
    # Player 1 Channels (Row 1)
    plot_channel_features(channels["obs_player_1"], axes[1, :], "P1-", frame_idx, n_columns, relic_weights=point_weights)
    # Player 1 Stats (Row 2)
    plot_stat_features(stats["episode_stats_player_0"], stats["episode_stats_player_1"], axes[2, :], "", frame_idx, n_columns)
    progressbar.update(1)

if __name__ == "__main__":

    config = parse_config()
    seed = np.random.randint(0, 100)
    print(f"Seed: {seed}")
    key = jax.random.PRNGKey(seed)
    vanilla_env = make_vanilla_env(env_args=config["env_args"], record=True, save_on_close=True, save_dir="test", save_format="html")
    vanilla_env = LogWrapper(vanilla_env, replace_info=True)
    env_params = sample_params(key)
    steps = 100

    channels_arrays, stats_arrays, point_weights = rollout(
        agent_0=JaxAgent("player_0", env_params.__dict__),
        agent_1=JaxAgent("player_1", env_params.__dict__),
        actor_0=NaiveAgent("player_0", env_params.__dict__),
        actor_1=NaiveAgent("player_1", env_params.__dict__),
        key=key,
        vanilla_env=vanilla_env,
        env_params=env_params,
        steps=steps
    )
    n_columns = max(len(channels_arrays["obs_player_0"].keys()) +1, len(stats_arrays["episode_stats_player_0"].keys()))
    fig, axes = plt.subplots(3, n_columns, figsize=(2*n_columns, 6))

    progressbar = tqdm(total=steps+1, desc="Generating GIF")

    anim = FuncAnimation(
        fig,
        lambda frame_idx: update(
            frame_idx, 
            channels=channels_arrays, 
            stats=stats_arrays, 
            progressbar=progressbar,
            n_columns = n_columns,
            point_weights = point_weights
        ),
        frames=steps,
        interval=250,
    )

    anim.save("purejarxrl_debug_channels.gif")
    plt.close(fig)
    print("GIF saved as 'purejarxrl_debug_channels.gif'")
