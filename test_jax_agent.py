
from purejaxrl.jax_agent import JaxAgent
from purejaxrl.make_env import make_env
from purejaxrl.utils import sample_params
import jax
from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env

env = LuxAIS3Env()
key = jax.random.PRNGKey(0)
env_params = sample_params(key)
obs, state = env.reset(key, env_params)
agent = JaxAgent('player_0', env_params.__dict__)
agent._actions(obs['player_0'], 60)