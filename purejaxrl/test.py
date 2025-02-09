import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from parse_config import parse_config

state_dict = parse_config("purejaxrl/jax_config_submission.yaml")["network"]["state_dict"]

print(state_dict["params"]["Dense_0"]["bias"][0:5], state_dict["batch_stats"]["SpectralNorm_0"]["spectral_norm/Conv_0/kernel/u"][0][0:5])