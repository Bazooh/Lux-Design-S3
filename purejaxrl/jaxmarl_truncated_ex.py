import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from jaxmarl.environments.overcooked import overcooked_layouts

import matplotlib.pyplot as plt


class CNN(nn.Module):
    activation: str =  "tanh" 
    @nn.compact
    def __call__(self, x):
        if self.activation ==  "relu" :
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str =  "tanh" 

    @nn.compact
    def __call__(self, x):
        if self.activation ==  "relu" :
            activation = nn.relu
        else:
            activation = nn.tanh
        
        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(
        LR: float =  0.0005,
        NUM_ENVS: int = 64,
        NUM_STEPS: int = 256,
        TOTAL_TIMESTEPS: int = 1e5,
        UPDATE_EPOCHS: int = 4,
        NUM_MINIBATCHES: int = 16,
        GAMMA: float = 0.99,
        GAE_LAMBDA: float = 0.95,
        CLIP_EPS: float = 0.2,
        ENT_COEF: float = 0.01,
        VF_COEF: float = 0.5,
        MAX_GRAD_NORM: float = 0.5,
        ACTIVATION: str = "relu" ,
        ENV_NAME: str =  "overcooked" ,
        ENV_KWARGS: dict =  {"layout" : overcooked_layouts["counter_circuit"]},
        ANNEAL_LR: bool = True,
        SEED: int = 0
):

    env = jaxmarl.make(ENV_NAME, **ENV_KWARGS)

    NUM_ACTORS = env.num_agents * NUM_ENVS
    NUM_UPDATES = (
        TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS
    )
    MINIBATCH_SIZE = (
        NUM_ACTORS * NUM_STEPS // NUM_MINIBATCHES
    )

    env = LogWrapper(env, replace_info=False)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (NUM_MINIBATCHES * UPDATE_EPOCHS))
            / NUM_UPDATES
        )
        return LR * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCritic(env.action_space().n, activation=ACTIVATION)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space().shape))

        network_params = network.init(_rng, init_x)
        if ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                optax.adam(LR, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, NUM_ENVS)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                jax.debug.print("last_obs {}", [last_obs[a] for a in env.agents])
                print("env.agents", env.agents)
                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)

                print("input_obs_shape", obs_batch.shape)

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, NUM_ENVS, env.num_agents
                )

                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, NUM_ENVS)

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                shaped_reward = info.pop("shaped_reward")
                current_timestep = update_step*NUM_STEPS*NUM_ENVS
                reward = jax.tree.map(lambda x,y: x+y, reward, shaped_reward)

                info = jax.tree.map(lambda x: x.reshape((NUM_ACTORS)), info)
                transition = Transition(
                    batchify(done, env.agents, NUM_ACTORS).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, NUM_ACTORS).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, NUM_STEPS
            )

            return runner_state, None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, NUM_UPDATES
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def main():
    SEED = 0
    rng = jax.random.PRNGKey(SEED)
    train_jit = jax.jit(make_train())
    out = train_jit(rng)

if __name__ == "__main__":
    main()