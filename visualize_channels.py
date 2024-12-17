import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agents.tensors.tensor import TensorConverter
from luxai_s3.wrappers import LuxAIS3GymEnv
import numpy as np
from rule_based.naive.naive_agent import NaiveAgent

observations_player0 = []
observations_player1 = []

# Initialize environment
env = LuxAIS3GymEnv()
observation, config = env.reset()

# Initialize agents
agent0 = NaiveAgent("player_0", config["params"])
agent1 = NaiveAgent("player_1", config["params"])

# Initialize TensorConverter
tensor_converter = TensorConverter()

# Collect observations
for _ in range(100):
    observations_player0.append(observation["player_0"])
    observations_player1.append(observation["player_1"])
    actions = {
        "player_0": agent0.actions(observation["player_0"]),
        "player_1": agent1.actions(observation["player_1"]),
    }
    observation, reward, terminated, truncated, info = env.step(actions)

# Prepare tensors for both players
tensors_player0 = []
tensors_player1 = []

for i in range(100):
    obs0 = observations_player0[i]
    obs1 = observations_player1[i]
    tensor0 = tensor_converter.convert(obs0, team_id=0)
    tensor1 = tensor_converter.convert(obs1, team_id=1)
    tensors_player0.append(tensor0)
    tensors_player1.append(tensor1)

tensors_player0 = np.stack(tensors_player0)
tensors_player1 = np.stack(tensors_player1)

assert tensors_player0.shape == (100, 23, 24, 24)
assert tensors_player1.shape == (100, 23, 24, 24)

# Set up the plot for both players (4 rows, 12 columns)
fig, axes = plt.subplots(4, 12, figsize=(20, 12))


# Function to plot tensor features
def plot_player_features(tensor, axes_rows, title_prefix):
    """Plots the tensor features across 2 rows of 12 subplots."""
    for i in range(23):
        row = axes_rows[0] if i < 12 else axes_rows[1]  # Top or bottom row
        col = i % 12  # Column index
        ax = row[col]
        ax.clear()
        ax.imshow(tensor[i], aspect="auto")
        ax.set_title(
            f"{title_prefix} {i}: {tensor_converter.channel_names[i]}", fontsize=6
        )
        ax.axis("off")


# Update function for animation
def update(frame):
    """Updates the plot for a given frame."""
    fig.suptitle(f"Frame {frame}", fontsize=16)
    # Player 0 (Rows 0-1)
    plot_player_features(tensors_player0[frame], axes[0:2], "P0 - Channel")
    # Player 1 (Rows 2-3)
    plot_player_features(tensors_player1[frame], axes[2:4], "P1 - Channel")


# Create animation
anim = FuncAnimation(fig, update, frames=len(tensors_player0), interval=100)

# Save the animation as a GIF
anim.save("tensor_features_players.gif")

plt.close(fig)

print("GIF saved as 'tensor_features_players.gif'")
