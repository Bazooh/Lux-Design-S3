import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agents.memory.memory import RelicMemory
from agents.tensor_converters.tensor import BasicMapExtractor
from luxai_s3.wrappers import LuxAIS3GymEnv
from rule_based.naive.naive_agent import NaiveAgent
from src.luxai_s3.env import Actions

# Initialize TensorConverter
tensor_converter = BasicMapExtractor()


def rollout():
    observations = [[], []]
    # Initialize environment
    env = LuxAIS3GymEnv()
    observation, config = env.reset()
    # Initialize agents
    agent0 = NaiveAgent("player_0", config["params"], RelicMemory())
    agent1 = NaiveAgent("player_1", config["params"], RelicMemory())

    # Collect observations
    for _ in range(100):
        observations[0].append(observation["player_0"])
        observations[1].append(observation["player_1"])
        actions: Actions = {
            "player_0": agent0.actions(observation["player_0"]),
            "player_1": agent1.actions(observation["player_1"]),
        }
        observation, reward, terminated, truncated, info = env.step(actions)
    return observations


tensors = [[], []]


def get_tensors(observations):
    for i in range(2):
        for obs in observations[i]:
            tensor = tensor_converter.convert(obs, i)
            tensors[i].append(tensor.cpu().numpy())
    return tensors


# Function to plot tensor features
def plot_player_features(tensor, axes_rows, title_prefix):
    """Plots the tensor features across 2 rows of 12 subplots."""
    for i in range(tensor.shape[0]):
        row = axes_rows[0] if i < 12 else axes_rows[1]  # Top or bottom row
        col = i % 12  # Column index
        ax = row[col]
        ax.clear()
        ax.imshow(tensor[i], cmap="coolwarm", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_title(
            f"{title_prefix} {i}: {tensor_converter.channel_names[i]}", fontsize=6
        )
        ax.axis("off")


fig, axes = plt.subplots(4, 12, figsize=(20, 12))


# Update function for animation
def update(frame):
    """Updates the plot for a given frame."""
    fig.suptitle(f"Frame {frame}", fontsize=16)
    # Player 0 (Rows 0-1)
    plot_player_features(tensors[0][frame], axes[0:2], "P0 - Channel")
    # Player 1 (Rows 2-3)
    plot_player_features(tensors[1][frame], axes[2:4], "P1 - Channel")


def main():
    global tensors

    observations = rollout()
    tensors = get_tensors(observations)


if __name__ == "__main__":
    main()

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(tensors[0]), interval=100)

    # Save the animation as a GIF
    anim.save("tensor_features_players.gif")

    plt.close(fig)

    print("GIF saved as 'tensor_features_players.gif'")
