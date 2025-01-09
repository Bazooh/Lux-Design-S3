import jax
import jax.numpy as jnp
import jax.nn as nn
from functools import partial

class MultiCategorical:
    def __init__(self, logits):
        """
        Args:
            logits: Array of logits of shape (batch_size, n_categories, n_values)
        """
        self.logits = logits


    def sample(self, key: jax.random.PRNGKey):
        """
        Sample from the multi-categorical distribution.
        
        Args:
            key: PRNGKey for sampling
        
        Returns:
            samples: Array of shape (batch_size,)
        """
        @partial(jax.vmap, in_axes=(0, 1))
        def single_cat_sampling(key, logits):
            probs = nn.softmax(logits, axis=-1)  # Apply softmax to get probabilities
            return jax.random.choice(key, a=logits.shape[-1], p=probs)
        
        action_keys = jax.random.split(key, self.logits.shape[0])  # Split key for batch
        return single_cat_sampling(action_keys, self.logits)

    def log_prob(self, values):
        """
        Calculate the log-probability of given values.
        
        Args:
            values: Array of sampled values (indices)
        
        Returns:
            log_probs: Array of log probabilities for each sample
        """
        probs = nn.softmax(self.logits, axis=-1)  # Apply softmax to get probabilities
        # Use the values to index the probabilities along the last axis (n_values)
        return jnp.log(jnp.take_along_axis(probs, values[..., None], axis=-1).squeeze(-1))

def gather_logits(logit_maps, row_indices, col_indices):
    # Gather the logits based on row_indices and col_indices
    logits_gathered_H = jnp.take_along_axis(logit_maps, row_indices[..., None], axis=1)  # Shape: (N, 16, W, 5)
    logits_gathered = jnp.take_along_axis(logits_gathered_H, col_indices[..., None], axis=2)  # Shape: (N, 16, 1, 5)
    logits_gathered = logits_gathered[:,:,0,:]  # Shape: (N, 16, 5)
    print("logits_gathered shape", logits_gathered.shape)
    return logits_gathered

# Example usage
logit_maps = jnp.ones((4, 16, 10, 5)) 
position = jnp.ones((1, 16, 2), dtype=jnp.int8)  
position = jnp.expand_dims(position, axis=-1)  # Shape: (N, 16, 2, 1)
# Step 3: Extract row and column indices from position
row_indices = position[:,:,0,:]  # Shape: (N, 16, 1)
col_indices = position[:,:,1,:]  # Shape: (N, 16, 1)

logits_gathered = gather_logits(logit_maps, row_indices, col_indices)

# Create the MultiCategorical distribution
pi = MultiCategorical(logits_gathered)

# Sample from the distribution
key = jax.random.PRNGKey(0)
samples = pi.sample(key)
print("Samples:", samples)

# Calculate log probabilities for given values
values = jnp.array([0, 1, 2, 3])  # Example values (indices)
log_probs = pi.log_prob(values)
print("Log probabilities:", log_probs)
