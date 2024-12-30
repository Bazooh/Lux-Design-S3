import jax
from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env
import numpy as np 
import time
from purejaxrl.sample_params import sample_params_fn
# the first env params is not batched and is used to initialize any static / unchaging values
# like map size, max units etc.
# note auto_reset=False for speed reasons. If True, the default jax code will attempt to reset each time and discard the reset if its not time to reset
# due to jax branching logic. It should be kept false and instead lax.scan followed by a reset after max episode steps should be used when possible since games
# can't end early.

# hyperparams
env = LuxAIS3Env(auto_reset=False, fixed_env_params=EnvParams())
num_envs = 128
seed = 0
np.random.seed(seed)
rng_key = jax.random.key(seed)


# define the vmapped functions 
reset_fn = jax.vmap(env.reset)
step_fn = jax.vmap(env.step)
action_space = (
    env.action_space()
) 

sample_action = jax.vmap(action_space.sample)

# sample random params initially
rng_key, subkey = jax.random.split(rng_key)
env_params = sample_params_fn(jax.random.split(subkey, num_envs))

obs, state = reset_fn(jax.random.split(subkey, num_envs), env_params)
obs, state, reward, terminated_dict, truncated_dict, info = step_fn(
    jax.random.split(subkey, num_envs),
    state,
    sample_action(jax.random.split(subkey, num_envs)),
    env_params,
)

max_episode_steps = (
    env.fixed_env_params.max_steps_in_match + 1
) * env.fixed_env_params.match_count_per_episode
rng_key, subkey = jax.random.split(rng_key)

@jax.jit
def random_rollout(rng_key, state, env_params):
    def take_step(carry, _):
        rng_key, state = carry
        rng_key, subkey = jax.random.split(rng_key)
        obs, state, reward, terminated_dict, truncated_dict, info = step_fn(
            jax.random.split(subkey, num_envs),
            state,
            sample_action(jax.random.split(subkey, num_envs)),
            env_params,
        )
        return (rng_key, state), (
            obs,
            state,
            reward,
            terminated_dict,
            truncated_dict,
            info,
        )

    _, (obs, state, reward, terminated_dict, truncated_dict, info) = jax.lax.scan(
        take_step, (rng_key, state), length=max_episode_steps, unroll=1
    )
    return obs, state, reward, terminated_dict, truncated_dict, info


if __name__ == "__main__":
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    st = time.time()
    random_rollout(subkey, state, env_params)

    print(f"FPS {max_episode_steps*num_envs/(time.time()-st)}")



