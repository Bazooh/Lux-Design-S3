import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agents.tensors.tensor import TensorConverter
from luxai_s3.state import EnvObs
from luxai_s3.wrappers import LuxAIS3GymEnv
import torch
import numpy as np
from rule_based.naive.naive_agent import NaiveAgent 
observations = []

env = LuxAIS3GymEnv()
observation, config = env.reset()

agent0 = NaiveAgent("player_0", config["params"])
agent1 = NaiveAgent("player_1", config["params"])

# Initialize TensorConverter
tensor_converter = TensorConverter()

for _ in range(100):
    observations.append(observation['player_0'])
    actions = {'player_0': agent0.actions(observation['player_0']),
                'player_1': agent1.actions(observation['player_1'])}
    observation, reward, terminated, truncated, info = env.step(actions)

# Prepare data for frames (frames 0 to 100)
tensors = []
for i in range(100):
    obs = observations[i]  # Get the ith observation
    team_id = 0  # Assuming team_id is 0; adjust if needed
    tensor = tensor_converter.convert(obs, team_id)
    tensors.append(tensor)
tensors = np.stack(tensors)

assert tensors.shape == (100, 23, 24, 24)

# Set up the plot
def plot_frame(tensor, ax):
    """Plots the given tensor feature maps on the provided Axes."""
    for i in range(23):

        row, col = divmod(i, 8)  # Determine row and column for subplot
        assert tensor[i].shape == (24, 24)
        ax[row, col].imshow(tensor[i], aspect='auto')
        ax[row, col].set_title(f"Channel {i}:"+ tensor_converter.channel_names[i], fontsize=8)
        ax[row, col].axis('off')

fig, axes = plt.subplots(3, 8, figsize=(16, 6))
fig.subplots_adjust(wspace=0.4, hspace=0.6)

# Update function for animation
def update(frame):
    """Updates the plot for a given frame."""
    for ax in axes.flat:
        ax.clear()  # Clear each subplot
    plot_frame(tensors[frame], axes)
    fig.suptitle(f"Frame {frame}", fontsize=16)

# Create animation
anim = FuncAnimation(fig, update, frames=len(tensors), interval=100)

# Save the animation as a GIF
anim.save("tensor_features.gif", writer="imagemagick")

plt.close(fig)

print("GIF saved as 'tensor_features.gif'")
