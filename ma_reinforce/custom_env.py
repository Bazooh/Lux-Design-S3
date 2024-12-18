from luxai_s3.wrappers import LuxAIS3GymEnv
import numpy as np
from utils import (
    is_alive_mask,
    relic_reward,
    sapped_reward,
    reveal_reward,
    high_energy_reward,
    death_reward,
)
# A custom environment to play 1 vs cpu.
# our AI will play as player 0
# the "opponent agent" will always play as player 1

class CustomLuxAIS3GymEnv():
    def __init__(self, opp_agent):
        self.opp_agent = opp_agent 
        self.env = LuxAIS3GymEnv()
        obs, config = self.env.reset()
        self.previous_obs = obs

    def get_reward_vector(self, previous_obs, actions, obs, reward, terminated, truncated, info):
        """
        The custom reward is a vector of rewards for each unit
        The individual reward of an unit is: 
            - reward for being close to relic
            - reward for succesful sapping
            - reward for revealing another player 
            - reward for revealing a relic
            - reward for high energy 
            - negative reward for death
        """
        total_revard_vector = np.zeros((self.env_cfg.max_units, 1), dtype=np.int32)

        # reward for being close to relic
        total_revard_vector += relic_reward(0, previous_obs, obs)

        # reward for sapping
        total_revard_vector += sapped_reward(0, previous_obs, obs, actions)

        # reward for revealing
        total_revard_vector += reveal_reward(0, previous_obs, obs)

        # reward for high energy
        total_revard_vector += high_energy_reward(0, previous_obs, obs)

        # negative reward for death
        total_revard_vector += death_reward(0, previous_obs, obs)

        # reward for revealed
        total_revard_vector += reveal_reward(0, previous_obs, obs)

        return 
    
    def get_done_vector(self, obs):
        done_vector = is_alive_mask(obs, 0)
        return done_vector

    def step(self, our_actions):
        """
        obs: dict[PlayerName, EnvObs]
        reward_vector: vector of rewards
        done_vector: vector of dones
        truncated: True if the match is over
        info: dict
        """
        opp_actions = self.opp_agent.actions(self.previous_obs["player_1"])
        actions = {"player_0": our_actions, "player_1": opp_actions}
        obs, reward, terminated, truncated, info = self.env.step(opp_actions)
        reward_vector = self.get_reward_vector(self.previous_obs, our_actions, obs, reward, terminated, truncated, info)
        done_vector = self.get_done_vector(obs)
        self.previous_obs = obs
        return obs, reward_vector, done_vector, truncated, info