import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from pytorch.env.wrappers import *
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from pytorch.torch_config import Args
import tyro
from tqdm import tqdm

def make_env(env_args, record=False, **record_kwargs):
    env = LuxAIS3GymEnv(numpy_output=True)
    if record:
        env = RecordEpisode(env, **record_kwargs)
    env = SimplifyTruncationWrapper(env)
    env = PointsMapWrapper(env)
    env = MemoryWrapper(env, env_args.memory)
    env = TransformActionWrapper(env, env_args.transform_action)
    env = TransformObsWrapper(env, env_args.transform_obs)
    return env

if __name__ == "__main__":
    args = tyro.cli(Args)
    env = make_env(args)
    obs, _ = env.reset()
        
    for i in tqdm(range(0, 510)):
        a = {
            "player_0": np.array([env.action_space().sample() for i in range(16)]), 
            "player_1": np.array([env.action_space().sample() for i in range(16)]), 
        }    
        # a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        if done:
            obs, _ = env.reset()
            print("reset")