import jax
import jax.numpy as jnp
def sample_action(key, logits):
    action = jax.random.categorical(key=key, logits=logits, axis=-1)  # Shape: (N, 16)
    return action

def get_logprob(logits, mask_awake, action):
    log_prob_group = jax.nn.log_softmax(logits, axis=-1)  # Shape: (N, 16, 5)
    log_prob_a = jnp.take_along_axis(log_prob_group, action[..., None], axis=-1).squeeze(axis=-1)  # Shape: (N, 16)
    log_prob_a_masked = log_prob_a * mask_awake  # Shape: (N, 16)
    log_prob= jnp.mean(log_prob_a_masked, axis=-1)/ jnp.sum(mask_awake, axis=-1)  # Shape: (N,)
    return(log_prob)

def get_entropy(logits):
    log_prob_group = jax.nn.log_softmax(logits, axis=-1)  # Shape: (N, 16, 5)
    entropy = -jnp.mean(jnp.sum(jnp.exp(log_prob_group) * log_prob_group, axis=-1), axis=-1)
    return(entropy)

# Parameters
N = 8
position = jnp.array([
    [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0],
    [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]
] * N).reshape(N, 16, 2)  # Shape: (N, 16, 2)
_rng = jax.random.PRNGKey(0)
logits = jax.random.normal(_rng, (N, 16, 5))  # Logits shape (N, 16, 5)


# Sample actions
action = sample_action(_rng, logits)

# Get log probabilities
mask_awake = (position[..., 0] >= 0).astype(jnp.float32)  # Shape: (N, 16), 1 if position >= 0 else 0
log_prob_a_masked = get_logprob(logits, mask_awake, action)
entropy = get_entropy(logits)

print("log_prob_a_masked shape:", log_prob_a_masked.shape)
print("log_prob_a_masked[0]:", log_prob_a_masked[0])
print("entropy shape:", entropy.shape)
print("entropy[0]:", entropy[0])