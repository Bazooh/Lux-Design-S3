
def is_alive_mask(obs, team_id):
    return obs.units_mask[team_id]

def relic_reward(team_id: int, previous_obs, obs):
    pass

def sapped_reward(team_id: int, previous_obs, obs, actions_of_team):
    pass 

def reveal_reward(team_id: int, previous_obs, obs):
    pass

def high_energy_reward(team_id: int, previous_obs, obs):
    pass

def death_reward(team_id: int, previous_obs, obs):
    pass
