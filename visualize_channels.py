from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm
from agents.tensor_converters.tensor import BasicTensorConverter
from rule_based.naive.naive_agent import NaiveAgent
from luxai_s3.env import Actions
from env_interface import RecordEpisode

tensor_converter0 = BasicTensorConverter()
tensor_converter1 = BasicTensorConverter()


def rollout(n_iter: int = 100):
    global tensor_converter

    numpy_tensors: list[list[np.ndarray]] = [[], []]
    # Initialize environment
    env = RecordEpisode("records")
    observation, env_params = env.reset(seed=randint(0, 1000))
    # Initialize agents
    agent0 = NaiveAgent("player_0", env_params)
    agent1 = NaiveAgent("player_1", env_params)

    # Collect observations
    for _ in range(n_iter):
        tensor_converter0.update_memory(observation.player_0, 0)
        tensor_converter1.update_memory(observation.player_1, 1)

        numpy_tensors[0].append(
            tensor_converter0.convert_channels(observation.player_0, 0)
        )
        numpy_tensors[1].append(
            tensor_converter1.convert_channels(observation.player_1, 1)
        )
        actions: Actions = {
            "player_0": agent0.actions(observation.player_0),
            "player_1": agent1.actions(observation.player_1),
        }
        observation, reward, terminated, truncated, info = env.step(actions)

    env.reset()  # Save the record

    return numpy_tensors


# Function to plot tensor features
def plot_player_features(
    numpy_tensors: np.ndarray, axes_rows: np.ndarray, title_prefix: str
):
    """Plots the tensor features across 2 rows of 12 subplots."""
    for i, channel_name in enumerate(tensor_converter0.channel_names()):
        row = axes_rows[0] if i < 12 else axes_rows[1]  # Top or bottom row
        col = i % 12  # Column index
        ax = row[col]
        ax.clear()
        ax.imshow(
            numpy_tensors[i].T,
            cmap="coolwarm",
            vmin=-0.5,
            vmax=0.5,
            aspect="auto",
            origin="upper",
        )
        ax.set_title(f"{title_prefix} {i}: {channel_name}", fontsize=6)
        ax.axis("off")


fig, axes = plt.subplots(4, 12, figsize=(20, 12))


# Update function for animation
def update(frame_idx: int, numpy_tensors: list[list[np.ndarray]], progress_bar: tqdm):
    """Updates the plot for a given frame."""
    fig.suptitle(f"Frame {frame_idx}", fontsize=16)
    # Player 0 (Rows 0-1)
    plot_player_features(numpy_tensors[0][frame_idx], axes[0:2], "P0 - Channel")
    # Player 1 (Rows 2-3)
    plot_player_features(numpy_tensors[1][frame_idx], axes[2:4], "P1 - Channel")

    progress_bar.update(1)


def main():
    numpy_tensors = rollout()
    n_frames = len(numpy_tensors[0])
    progressbar = tqdm(total=n_frames + 1)

    anim = FuncAnimation(
        fig,
        lambda frame_idx, numpy_tensors=numpy_tensors, progressbar=progressbar: update(
            frame_idx, numpy_tensors, progressbar
        ),  # type: ignore
        frames=n_frames,
        interval=100,
    )

    anim.save("tensor_features_players.gif")

    plt.close(fig)

    print("GIF saved as 'tensor_features_players.gif'")


if __name__ == "__main__":
    main()
