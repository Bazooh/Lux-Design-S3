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
from agents.reward_shapers.reward import (
    DistanceToNearestRelicRewardShaper,
    GreedyExploreRewardShaper,
    GreedyRewardShaper,
    RewardShaper,
)
from agents.tensor_converters.tensor import BasicTensorConverter
from luxai_s3.wrappers import RecordEpisode, Actions, PlayerAction, PlayerReward

from agents.rl_agent import BasicRLAgent
from agents.vec_rl_agent import VecBasicRLAgent
from rule_based.naive.naive_agent import NaiveAgent

import time
from env_interface import (
    EnvInterface,
    PlayerAgentMask,
    VecEnvInterface,
    VecPlayerAction,
    VecPlayerReward,
    VecPlayerAgentMask,
)

from config import TRAINING_DEVICE, SAMPLING_DEVICE

PROFILE = False  # if enabled, profiles the code
USE_WANDB = True  # if enabled, logs data on wandb server


class ReplayBuffer:
    def __init__(self, buffer_limit: int):
        self.buffer: collections.deque[
            tuple[
                torch.Tensor,
                PlayerAction,
                PlayerReward,
                torch.Tensor,
                PlayerAgentMask,
                PlayerAgentMask,
            ]
        ] = collections.deque(maxlen=buffer_limit)

    def put(
        self,
        obs: torch.Tensor,
        actions: VecPlayerAction,
        reward: VecPlayerReward,
        next_obs: torch.Tensor,
        done: VecPlayerAgentMask,
        awake: VecPlayerAgentMask,
    ):
        size = obs.shape[0]
        for player_id in range(2):
            for i in range(size):
                self.buffer.append(
                    (
                        obs[i, player_id],
                        actions[i, player_id],
                        reward[i, player_id],
                        next_obs[i, player_id],
                        done[i, player_id],
                        awake[i, player_id],
                    )
                )

    def sample(self, n: int):
        """obs, actions, reward, next_obs, done, awake\n
        Awake and Done maks are True when the unit is alive"""
        mini_batch = random.sample(self.buffer, n)
        s_tensor = torch.empty((n, *mini_batch[0][0].shape), dtype=torch.float)
        a_tensor = torch.empty((n, *mini_batch[0][1].shape), dtype=torch.float)
        r_tensor = torch.empty((n, *mini_batch[0][2].shape), dtype=torch.float)
        s_prime_tensor = torch.empty((n, *mini_batch[0][3].shape), dtype=torch.float)
        done_mask_tensor = torch.empty((n, *mini_batch[0][4].shape), dtype=torch.float)
        awake_mask_tensor = torch.empty((n, *mini_batch[0][5].shape), dtype=torch.float)

        for i, (s, a, r, s_prime, done, awake) in enumerate(mini_batch):
            s_tensor[i] = s
            a_tensor[i] = torch.from_numpy(a)
            r_tensor[i] = torch.from_numpy(r)
            s_prime_tensor[i] = s_prime
            done_mask_tensor[i] = torch.from_numpy(done)
            awake_mask_tensor[i] = torch.from_numpy(awake)

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
    q.to(TRAINING_DEVICE)

    train_loss = 0

    for _ in range(update_iter):
        s, a, r, s_prime, done_mask, awake_mask = memory.sample(batch_size)

        q_out: torch.Tensor = q(s.to(TRAINING_DEVICE)).cpu()

        q_a = q_out.gather(2, a[:, :, 0].unsqueeze(-1).long()).squeeze(-1)
        max_q_prime = q_target(s_prime.to(TRAINING_DEVICE)).cpu().max(dim=2).values

        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a * awake_mask, target.detach() * awake_mask)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    q.to(SAMPLING_DEVICE)

    return train_loss / update_iter


def test(
    env: EnvInterface | RecordEpisode,
    num_episodes: int,
    network: nn.Module,
    agent_instantiator: Callable[..., BasicRLAgent],
) -> float:
    score: float = 0

    for episode_i in range(num_episodes):
        obs, env_params = env.reset()
        agent_0 = agent_instantiator("player_0", env_params, SAMPLING_DEVICE, network)
        agent_1 = NaiveAgent("player_1", env_params)

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

    return score / (num_episodes if num_episodes != 0 else 1)


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
    n_envs: int,
    network_instantiator: Callable[[], nn.Module],
    agent_instantiator: Callable[..., VecBasicRLAgent],
    test_agent_instantiator: Callable[..., BasicRLAgent],
    reward_shaper: RewardShaper,
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

    env = VecEnvInterface(n_envs, BasicTensorConverter, reward_shaper)

    network = network_instantiator().to(SAMPLING_DEVICE)
    if resume_path is not None:
        network.load_state_dict(torch.load(resume_path))
    network_target = network_instantiator().to(TRAINING_DEVICE)
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
        obs, env_params = env.reset()
        agent = agent_instantiator(
            env_params,
            device=SAMPLING_DEVICE,
            model=network,
        )

        obs, _, _, _, info = env.step(np.zeros((n_envs, 2, 16, 3), dtype=np.int32))

        simulation_score: float = 0
        game_finished = False
        count_frames: int = 0
        start_time = time.time()
        while not game_finished:
            count_frames += 1

            actions = (
                agent.sample_actions(obs.view(n_envs * 2, -1, 24, 24), epsilon)
                .view(n_envs, 2, 16, 3)
                .numpy()
            )

            next_obs, rewards, awake_mask, done_mask, info = env.step(actions)
            game_finished = info["game_finished"].all()

            memory.put(obs, actions, rewards, next_obs, done_mask, awake_mask)

            simulation_score += (rewards * awake_mask)[:, 0].mean().item()
            obs = next_obs

        score += simulation_score / count_frames

        if memory.size() > warm_up_steps:
            train_loss = train(
                network,
                network_target,
                memory,
                optimizer,
                gamma,
                batch_size,
                update_iter,
            )

        fps.append(count_frames * n_envs / (time.time() - start_time))

        # EVALUATION
        if episode_i % log_interval == 0 and episode_i != 0:
            network_target.load_state_dict(network.state_dict())
            torch.save(network.state_dict(), f"models_weights/network_{episode_i}.pth")

            test_score = test(test_env, test_episodes, network, test_agent_instantiator)
            print(
                f"#{episode_i:<10}/{max_episodes} episodes, avg train score : {score / log_interval:.2f}, test score: {test_score:.2f}, fps : {int(np.mean(fps))}, n_buffer : {memory.size()}, eps : {epsilon:.1f}"
            )
            if USE_WANDB:
                wandb.log(
                    {
                        "episode": episode_i,
                        "test-score": test_score,
                        "buffer-size": memory.size(),
                        "epsilon": epsilon,
                        "train-score": score / log_interval,
                        "fps": np.mean(fps),
                        "loss": train_loss if memory.size() > warm_up_steps else 0,  # type: ignore
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
        "log_interval": 50,
        "max_episodes": 4000,
        "max_epsilon": 0.9,
        "min_epsilon": 0.1,
        "test_episodes": 0,
        "warm_up_steps": 2000,
        "update_iter": 20,
        "n_envs": 4,
        "network_instantiator": lambda: CNN(n_input_channels=23),
        "agent_instantiator": VecBasicRLAgent,
        "test_agent_instantiator": BasicRLAgent,
        "reward_shaper": GreedyRewardShaper()
        + 0.1 * DistanceToNearestRelicRewardShaper()
        + 0.01 * GreedyExploreRewardShaper(),
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
