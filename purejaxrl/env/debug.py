from math import e
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import jax, chex
from typing import Any
from purejaxrl.env.utils import sample_params
from luxai_s3.env import EnvObs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from purejaxrl.purejaxrl_agent import RawPureJaxRLAgent, PureJaxRLAgent
from tqdm import tqdm
from purejaxrl.env.make_env import make_vanilla_env, TrackerWrapper, LogWrapper
from purejaxrl.parse_config import parse_config
from rule_based_jax.naive.agent import NaiveAgent_Jax
from purejaxrl.eval_jax import run_episode_and_record

# Function to plot tensor features
def plot_channel_features(channels: dict, axes_row: np.ndarray, title_prefix: str, frame_idx: int, n_columns: int, points_map):
    for i, (feat_name, feat_array) in enumerate(channels.items()):
        ax = axes_row[i]
        ax.clear()
        if feat_name in ["Ally_Units", "Enemy_Units"]: 
            ax.imshow(feat_array[frame_idx].T, cmap = "bwr", vmin = -2, vmax = 2, aspect = "auto")
        if feat_name in ["Points", "Relic"]: 
            ax.imshow(feat_array[frame_idx].T, cmap = "bwr", vmin = -1, vmax = 1, aspect = "auto")
        else:
            ax.imshow(feat_array[frame_idx].T, cmap = "bwr", aspect = "auto")
        ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 24, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.3)
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        ax.set_title(f"{title_prefix} {feat_name}", fontsize=10)
        

    # Plot relic weights
    ax = axes_row[i+1]
    ax.clear()
    ax.imshow(points_map[frame_idx].T, cmap = "bwr", vmin = -1, vmax = 1, aspect = "auto")
    ax.set_title(f"(Ground Truth) Points", fontsize=10)
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 24, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.3)
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    for i in range(len(channels.items())+1, n_columns):
        ax = axes_row[i]
        ax.clear()

# Function to plot tensor features
def plot_vector_features(vectors: dict, axes_row: np.ndarray, title_prefix: str, frame_idx: int):
    for i, (feat_name, feat_array) in enumerate(vectors.items()):
        ax = axes_row[i]
        ax.clear()
        ax.imshow([[feat_array[frame_idx]]], cmap="bwr", vmin=-2, vmax=2, aspect="auto")
        ax.text(0, 0, f"{feat_array[frame_idx]:.2f}", fontsize=14)
        ax.set_title(f"{title_prefix} {feat_name}", fontsize=10)
    
    
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
            vectors: dict,
            stats: dict,
            points_map,
            n_columns, 
            progressbar):
    # Player 0 Channels (Row 0)
    plot_channel_features(channels["obs_player_0"], axes[0, :], "P0-", frame_idx, n_columns, points_map=points_map)
    # Player 1 Channels (Row 1)
    plot_vector_features(vectors["obs_player_0"], axes[1, :], "P0-", frame_idx)
    # Player 1 Channels (Row 2)
    plot_channel_features(channels["obs_player_1"], axes[2, :], "P1-", frame_idx, n_columns, points_map=points_map)
    # Player 2 Channels (Row 3)
    plot_vector_features(vectors["obs_player_1"], axes[3, :], "P1-", frame_idx)
    # Player 1 Stats (Row 4)
    plot_stat_features(stats["episode_stats_player_0"], stats["episode_stats_player_1"], axes[4, :], "", frame_idx, n_columns)
    progressbar.update(1)

if __name__ == "__main__":

    config = parse_config()
    seed = np.random.randint(0, 100)
    print(f"Seed: {seed}")
    key = jax.random.PRNGKey(seed)
    rec_env = make_vanilla_env(env_args=config["env_args"], record=True, save_on_close=True, save_dir=".", save_format="html")
    rec_env = LogWrapper(rec_env, replace_info=True)
    
    agent_0=PureJaxRLAgent("player_0")
    agent_1=NaiveAgent_Jax("player_1")
    
    channels_arrays, vec_arrays, stats_arrays, points_map = run_episode_and_record(
        rec_env = rec_env,
        agent_0 = agent_0,
        agent_1 = agent_1, 
        key = key, 
        match_count_per_episode = 1,
        use_tdqm = True,
        return_states = True,
        plot_stats_curves = False        
    )
    
    steps = len(points_map)
    n_columns = max(len(channels_arrays["obs_player_0"].keys()) +1, max(len(stats_arrays["episode_stats_player_0"].keys()), len(vec_arrays["obs_player_0"].keys())))
    fig, axes = plt.subplots(5, n_columns, figsize=(3*n_columns, 20))

    progressbar = tqdm(total=steps+1, desc="Generating MP4")

    anim = FuncAnimation(
        fig,
        lambda frame_idx: update(
            frame_idx, 
            channels=channels_arrays, 
            vectors=vec_arrays,
            stats=stats_arrays, 
            progressbar=progressbar,
            n_columns = n_columns,
            points_map = points_map
        ),
        frames=steps,
        interval=250,
    )

    anim.save("purejarxrl_debug_channels.mp4", writer="ffmpeg", fps=4)
    plt.close(fig)
    print("GIF saved as 'purejarxrl_debug_channels.mp4'")
