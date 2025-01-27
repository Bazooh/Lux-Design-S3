
from typing import Any
import shutil
import jax
from jax import numpy as jnp
from flax import linen as nn
import orbax.checkpoint
import optax
import os
import jax
import optax
from flax.training import orbax_utils
from utils import init_state_dict, CustomTrainState

####################### INITIALIZE ######################################
from env.make_env import make_env
from parse_config import parse_config

config = parse_config()
env = make_env(config["env_args"])
model = config["network"]["model"]
x = env.observation_space.sample(jax.random.PRNGKey(0))
expanded_x = {x: jnp.expand_dims(value, axis=0) for x, value in x.items()}
y = jnp.ones((1,))


#######################  TRAINING ######################################
optimizer = optax.sgd(learning_rate=0.1)
state_dict = init_state_dict(model = model, key = jax.random.PRNGKey(0), init_x = x)

state = CustomTrainState.create(
    apply_fn=model.apply,
    params=state_dict["params"],
    tx=optimizer,
    batch_stats=state_dict["batch_stats"],
)


   
def train_step(state, x, y):
    def loss_fn(params):
        (_, value, _), updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            **expanded_x,
            train=True,
            mutable=['batch_stats'],
        )
        loss = jnp.mean(optax.l2_loss(predictions=value, targets=y))
        return loss, updates

    (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(
        grads=grads, batch_stats=updates['batch_stats']
    )
    return new_state, loss

# Training loop
for epoch in range(10):
    state, loss = train_step(state, x, y)

####################### AFTER TRAINING ######################################

ckpt_dir = '/home/hadriencrs/Code/python/Lux-Design-S3/tmp'

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)  # Remove any existing checkpoints from the last notebook run.

ckpt = {'params': state.params, 'batch_stats': state.batch_stats}

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save('/home/hadriencrs/Code/python/Lux-Design-S3/tmp/single_save', ckpt, save_args=save_args)

options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    '/home/hadriencrs/Code/python/Lux-Design-S3/tmp/managed', orbax_checkpointer, options)

restored_state = orbax_checkpointer.restore('/home/hadriencrs/Code/python/Lux-Design-S3/tmp/single_save')

# print(restored_state["params"])
# print(restored_state["batch_stats"])

_, value, _ = model.apply(restored_state, **expanded_x , train=False)
print(value, restored_state["batch_stats"])
