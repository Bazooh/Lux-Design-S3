import collections
import random
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from agents.models.dense import CNN
from luxai_s3.wrappers import RecordEpisode, PlayerAction, Actions
from agents.rl_agent import BasicRLAgent
from rule_based.naive.naive_agent import NaiveAgent
from agents.reward_shapers.reward import Reward

import time
from env_interface import EnvInterface

PROFILE = False  # if enabled, profiles the code
USE_WANDB = False  # if enabled, logs data on wandb server


class ReplayBuffer:
    def __init__(self, buffer_limit: int):
        self.buffer: collections.deque[
            tuple[
                torch.Tensor,
                PlayerAction,
                Reward,
                torch.Tensor,
                np.ndarray[Literal[16], np.dtype[np.bool_]],
                np.ndarray[Literal[16], np.dtype[np.bool_]],
            ]
        ] = collections.deque(maxlen=buffer_limit)

    def put(
        self,
        obs: torch.Tensor,
        actions: PlayerAction,
        reward: Reward,
        next_obs: torch.Tensor,
        done: np.ndarray[Literal[16], np.dtype[np.bool_]],
        awake: np.ndarray[Literal[16], np.dtype[np.bool_]],
    ):
        self.buffer.append((obs, actions, reward, next_obs, done, awake))

    def sample(self, n: int):
        """obs, actions, reward, next_obs, done"""
        mini_batch = random.sample(self.buffer, n)
        s_tensor = torch.empty((n, *mini_batch[0][0].shape), dtype=torch.float)
        a_tensor = torch.empty((n, *mini_batch[0][1].shape), dtype=torch.float)
        r_tensor = torch.empty((n, *mini_batch[0][2].shape), dtype=torch.float)
        s_prime_tensor = torch.empty((n, *mini_batch[0][3].shape), dtype=torch.float)
        done_mask_tensor = torch.empty((n, *mini_batch[0][4].shape), dtype=torch.float)
        awake_mask_tensor = torch.empty((n, *mini_batch[0][5].shape), dtype=torch.float)

        for i, (s, a, r, s_prime, done, awake) in enumerate(mini_batch):
            s_tensor[i] = s
            a_tensor[i] = torch.tensor(a)
            r_tensor[i] = torch.tensor(r)
            s_prime_tensor[i] = s_prime
            done_mask_tensor[i] = torch.tensor(done)
            awake_mask_tensor[i] = torch.tensor(awake)

        return (
            s_tensor,
            a_tensor,
            r_tensor,
            s_prime_tensor,
            done_mask_tensor,
            awake_mask_tensor,
        )

    def size(self):
        return len(self.buffer)


def train(
    q: nn.Module,
    q_target: nn.Module,
    memory: ReplayBuffer,
    optimizer: optim.Optimizer,
    gamma: float,
    batch_size: int,
    update_iter: int = 10,
):
    for _ in range(update_iter):
        s, a, r, s_prime, done_mask, awake_mask = memory.sample(batch_size)

        q_out: torch.Tensor = q(s)

        q_a = q_out.gather(2, a[:, :, 0].unsqueeze(-1).long()).squeeze(-1)
        max_q_prime = q_target(s_prime).max(dim=2)[0]

        target = r * awake_mask + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a * awake_mask, target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(
    env: EnvInterface | RecordEpisode, num_episodes: int, network: nn.Module
) -> float:
    score: float = 0

    for episode_i in range(num_episodes):
        obs, config = env.reset()
        agent_0 = BasicRLAgent("player_0", config["params"], network)
        agent_1 = NaiveAgent("player_1", config["params"])

        game_finished = False
        while not game_finished:
            actions: Actions = {
                "player_0": agent_0.actions(obs.player_0),
                "player_1": agent_1.actions(obs.player_1),
            }
            next_obs, reward, _, truncated, _ = env.step(actions)
            game_finished = truncated["player_0"].item() or truncated["player_1"].item()

            score += reward["player_0"].item() - reward["player_1"].item()
            obs = next_obs

    return score / num_episodes


def main(
    lr: float,
    gamma: float,
    batch_size: int,
    buffer_limit: int,
    log_interval: int,
    max_episodes: int,
    max_epsilon: float,
    min_epsilon: float,
    test_episodes: int,
    warm_up_steps: int,
    update_iter: int,
    network_instantiator: Callable[[], nn.Module],
    monitor: bool = False,
    save_format: Literal["json", "html"] = "json",
    resume_path: str | None = None,
    resume_iter: int | None = None,
):
    assert (
        resume_path is None or resume_iter is not None
    ), "resume_iter must be provided if resume_path is provided"
    assert (
        resume_iter is None or resume_path is not None
    ), "resume_path must be provided if resume_iter is provided"

    env = EnvInterface()
    network = network_instantiator()
    if resume_path is not None:
        network.load_state_dict(torch.load(resume_path))
    network_target = network_instantiator()
    network_target.load_state_dict(network.state_dict())

    test_env = EnvInterface()
    if monitor:
        test_env = RecordEpisode(
            test_env,
            save_dir="records",
            save_format=save_format,
        )
    memory = ReplayBuffer(buffer_limit)

    optimizer = optim.Adam(network.parameters(), lr=lr)

    score: float = 0
    fps = []

    for episode_i in tqdm(
        range(0 if resume_iter is None else resume_iter, max_episodes)
    ):
        epsilon = max(
            min_epsilon,
            max_epsilon
            - (max_epsilon - min_epsilon) * (episode_i / (0.4 * max_episodes)),
        )
        obs, config = env.reset()
        agent_0 = BasicRLAgent("player_0", config["params"], network)
        agent_1 = BasicRLAgent("player_1", config["params"], network)

        obs, _, _, _, _ = env.step(
            {
                "player_0": np.zeros((16, 3), dtype=np.int32),
                "player_1": np.zeros((16, 3), dtype=np.int32),
            }
        )
        obs_tensor_0 = agent_0.obs_to_tensor(obs.player_0)
        obs_tensor_1 = agent_1.obs_to_tensor(obs.player_1)

        agent_0.update_obs(obs.player_0)
        agent_1.update_obs(obs.player_1)

        simulation_score: float = 0
        game_finished = False
        count_frames, start_time = 0, time.time()
        while not game_finished:
            count_frames += 1

            actions: Actions = {
                "player_0": agent_0.sample_action(obs_tensor_0, epsilon),
                "player_1": agent_1.sample_action(obs_tensor_1, epsilon),
            }
            next_obs, reward, _, truncated, _ = env.step(
                {
                    "player_0": actions["player_0"],
                    "player_1": agent_1.symetric_action(actions["player_1"]),
                }
            )
            game_finished = truncated["player_0"].item() or truncated["player_1"].item()

            agent_0.update_obs(next_obs.player_0)
            agent_1.update_obs(next_obs.player_1)

            next_obs_tensor_0 = agent_0.obs_to_tensor(next_obs.player_0)
            next_obs_tensor_1 = agent_1.obs_to_tensor(next_obs.player_1)

            reward_0 = agent_0.reward_shaper.convert(
                obs.player_0,
                obs_tensor_0,
                reward["player_0"].item(),
                actions["player_0"],
                next_obs.player_0,
                next_obs_tensor_0,
                0,
            )
            reward_1 = agent_1.reward_shaper.convert(
                obs.player_1,
                obs_tensor_1,
                reward["player_1"].item(),
                actions["player_1"],
                next_obs.player_1,
                next_obs_tensor_1,
                1,
            )

            awake_mask_0 = np.array(obs.player_0.units_mask[0])

            memory.put(
                obs_tensor_0,
                actions["player_0"],
                reward_0,
                next_obs_tensor_0,
                np.array(next_obs.player_0.units_mask[0]),
                awake_mask_0,
            )
            memory.put(
                obs_tensor_1,
                actions["player_1"],
                reward_1,
                next_obs_tensor_1,
                np.array(next_obs.player_1.units_mask[1]),
                np.array(obs.player_1.units_mask[1]),
            )

            simulation_score += (reward_0 * awake_mask_0).sum().item()

            obs_tensor_0 = next_obs_tensor_0
            obs_tensor_1 = next_obs_tensor_1
            obs = next_obs

        score += simulation_score / obs.player_0.steps

        if memory.size() > warm_up_steps:
            train(
                network,
                network_target,
                memory,
                optimizer,
                gamma,
                batch_size,
                update_iter,
            )

        fps.append(count_frames / (time.time() - start_time))

        # EVALUATION

        if episode_i % log_interval == 0 and episode_i != 0:
            network_target.load_state_dict(network.state_dict())
            torch.save(network.state_dict(), f"models_weights/network_{episode_i}.pth")

            test_score = test(test_env, test_episodes, network)
            print(
                f"#{episode_i:<10}/{max_episodes} episodes, avg train score : {score / log_interval:.1f}, test score: {test_score:.1f}, fps : {np.mean(fps):.1f}, n_buffer : {memory.size()}, eps : {epsilon:.1f}"
            )
            if USE_WANDB:
                wandb.log(
                    {
                        "episode": episode_i,
                        "test-score": test_score,
                        "buffer-size": memory.size(),
                        "epsilon": epsilon,
                        "train-score": score / log_interval,
                    }
                )
            score = 0

    env.close()
    test_env.close()


if __name__ == "__main__":
    kwargs = {
        "lr": 0.0005,
        "batch_size": 32,
        "gamma": 0.99,
        "buffer_limit": 50000,
        "log_interval": 20,
        "max_episodes": 10000,
        "max_epsilon": 0.9,
        "min_epsilon": 0.1,
        "test_episodes": 5,
        "warm_up_steps": 2000,
        "update_iter": 10,
        "network_instantiator": CNN,
    }
    if USE_WANDB:
        import wandb

        wandb.init(
            project="minimal-marl", config={"algo": "idqn", **kwargs}, monitor_gym=True
        )

    if PROFILE:
        import cProfile

        cProfile.run("main(**kwargs)", filename="profile.prof")

    else:
        main(
            monitor=True,
            save_format="html",
            # resume_path="models_weights/network_1000.pth",
            # resume_iter=1000,
            **kwargs,
        )
