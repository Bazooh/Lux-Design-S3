import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import random
import time
from pytorch.env.make_env import make_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from pytorch.network import Pix2Pix_AC
from torch_config import Args
from tqdm import tqdm

def make_env_g(idx, env_args):
    def thunk():
        return make_env(env_args)
    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = "pytorch_ppo"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [lambda : make_env(args) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Pix2Pix_AC(
        action_dim=envs.single_action_space.n, 
        action_masking=args.action_masking, 
        normalize_value=args.normalize_value,
        n_channels=args.n_channels,
        n_resblocks=args.n_resblocks,
        embedding_time=args.embedding_time,
        normalize_logits=args.normalize_logits
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    OBS_KEYS = list(envs.single_observation_space["player_0"].keys())
    # ALGO Logic: Storage setup
    obs0_v = {k: torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space["player_0"][k].shape).to(device) for k in OBS_KEYS}
    actions0_v = torch.zeros((args.num_steps, args.num_envs) + (16, 6)).to(device)
    logprobs0_v = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards0_v = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_v = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values0_v = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_v, _ = envs.reset(seed=args.seed)
    next_obs0_v, next_obs1_v = next_obs_v.player_0, next_obs_v.player_0
    next_obs0_v, next_obs1_v = {k: torch.Tensor(v).to(device) for k, v in next_obs0_v.items() }, {k: torch.Tensor(v).to(device) for k, v in next_obs1_v.items() }
    next_done_v = torch.zeros(args.num_envs).to(device)

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            for k in OBS_KEYS: obs0_v[k][step] = next_obs0_v[k]
            dones_v[step] = next_done_v

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action0_v, logprob0_v, _, value0_v = agent.get_action_and_value(next_obs0_v)
                values0_v[step] = value0_v.flatten()
                action1_v, _, _, _ = agent.get_action_and_value(next_obs1_v)
            actions0_v[step] = action0_v
            logprobs0_v[step] = logprob0_v

            # TRY NOT TO MODIFY: execute the game and log data.
            action = {"player_0": action0_v.cpu().numpy(), "player_1": action1_v.cpu().numpy()}
            next_obs_v, reward_v, terminations, truncations, infos = envs.step(action)
            next_obs0_v, next_obs1_v = next_obs_v.player_0, next_obs_v.player_0
            next_obs0_v, next_obs1_v = {k: torch.Tensor(v).to(device) for k, v in next_obs0_v.items() }, {k: torch.Tensor(v).to(device) for k, v in next_obs1_v.items() }
            reward_v = np.array(reward_v["player_0"])
            next_done = np.logical_or(terminations, truncations)
            rewards0_v[step] = torch.tensor(reward_v).to(device).view(-1)
            next_obs = {k: torch.Tensor(v).to(device) for k, v in next_obs.items() }
            next_obs0_v, next_obs1_v = next_obs_v["player_0"], next_obs_v["player_1"]
            next_obs0_v, next_obs1_v = torch.Tensor(next_obs0_v).to(device), torch.Tensor(next_obs1_v).to(device)
            next_done = torch.Tensor(next_done).to(device)

            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards0_v).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_v[t + 1]
                    nextvalues = values0_v[t + 1]
                delta = rewards0_v[t] + args.gamma * nextvalues * nextnonterminal - values0_v[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values0_v

        # flatten the batch
        b_obs = {next_obs0_v[k].reshape((-1,) + envs.single_observation_space.shape) for k in OBS_KEYS}
        b_logprobs = logprobs0_v.reshape(-1)
        b_actions = actions0_v.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values0_v.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


    envs.close()