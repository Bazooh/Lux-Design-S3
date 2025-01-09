import jax.numpy as jnp
import jax
import flax.linen as nn
from functools import partial
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer
    """
    channel: int
    reduction: int

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        # Squeeze: Global Average Pooling
        y = jnp.mean(x, axis=(1, 2), keepdims=False)  # Shape: (b, c)
        # Excitation: Fully Connected layers with reduction
        y = nn.Dense(self.channel // self.reduction)(y)
        y = nn.relu(y)
        y = nn.Dense(self.channel)(y)
        y = nn.sigmoid(y)  # Shape: (b, c)
        # Reshape for scaling
        y = jnp.expand_dims(jnp.expand_dims(y, axis=1), axis=1)  # Shape: (b, 1, 1, c)
        # Scale: Channel-wise multiplication
        return x * y

class ResidualBlock(nn.Module):
    """
    Residual Block:
                x
             /      \  
            /        \ 
           V          V 
    conv + se-layer    shortcut
           \          / 
            \        /  
             V   +  V
                out

    """
    in_channel: int
    out_channel: int
    kernel_size: int 
    padding: int 
    
    @nn.compact
    def __call__(self, x):
        conv_block = nn.Sequential([
            nn.Conv(self.out_channel, kernel_size=self.kernel_size, strides=1, padding=self.padding),
            nn.leaky_relu,
        ])# conv block
        se_layer = SELayer(self.out_channel, reduction=16)
        out = x + se_layer(conv_block(x))
        out = nn.leaky_relu(out)
        return out

class HybridActorCritic(nn.Module):
    """
                    Image (C,24,24)
                        |
                        | Transpose
                        V
                    Image (24,24,C)                    
                        |
                        | 1x1-Conv
                        V
                    Image (24,24,32)
                        |
                        | ResBlock 1
                        V
                    Image'(24,24,32)
                    /       \  
        ResBlock2  /         \  Mean + Value Head (1)
                  /           \ 
                 V             V
        Image''(24,24,32)     Output Value
                |
    1x1-Conv    |
                V
        Logit Maps(24,24,5)
                |
    Pos-Masking |
                V
            Logits (16,5)
    """
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, image, vector, position):
        conv_block_0 = nn.Sequential([
            nn.Conv(32, kernel_size=1, strides=1, padding=0),
            nn.leaky_relu,
        ]) # conv block 
        conv_block_1 = nn.Sequential([
            nn.Conv(self.action_dim, kernel_size=1, strides=1, padding=0),
            nn.leaky_relu,
        ]) # conv block 
        res_block_1 = ResidualBlock(in_channel=32, out_channel=32, kernel_size=5, padding=2) 
        res_block_2 = ResidualBlock(in_channel=32, out_channel=32, kernel_size=5, padding=2)
        value_head = nn.Sequential([
            nn.Dense(32),
            nn.leaky_relu,
            nn.Dense(32),
            nn.leaky_relu,
            nn.Dense(1),
        ])

        image = image.transpose(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        x = res_block_1(conv_block_0(image))
        value = value_head(x.mean(3).mean(2))
        value = jnp.squeeze(value, axis=-1)
        
        logit_maps = conv_block_1(res_block_2(x))
        position = jnp.expand_dims(position, axis=-1)  # Shape: (N, 16, 2, 1)

        # Step 3: Extract row and column indices from position
        row_indices = position[:,:,0,:]  # Shape: (N, 16, 1)
        col_indices = position[:,:,1,:]  # Shape: (N, 16, 1)

        #  Gather 
        logits_gathered_H = jnp.take_along_axis(logit_maps, row_indices[..., None], axis=1)  # Shape: (N, 16, W, 5)
        logits_gathered = jnp.take_along_axis(logits_gathered_H, col_indices[..., None], axis=2)  # Shape: (N, 16, 1, 5)
        logits_gathered = logits_gathered[:,:,0,:]  # Shape: (N, 16, 5)
        print("logits_gathered shape", logits_gathered.shape)
        pi = MultiCategorical(logits_gathered)
        return pi, value
    
class MultiCategorical:
    def __init__(self, logits):
        """
        Args:
            logits: Array of logits of shape (batch_size, n_categories, n_values)
        """
        self.logits = logits

    @partial(jax.vmap, in_axes=(0, None))
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