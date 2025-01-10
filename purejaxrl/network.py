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
                    Image (B,C,24,24)
                        |
                        | Transpose
                        V
                    Image (B,24,24,C)                    
                        |
                        | 1x1-Conv
                        V
                    Image (B,24,24,32)
                        |
                        | ResBlock 1
                        V
                    Image'(B,24,24,32)
                    /       \  
        ResBlock2  /         \  Mean + Value Head (B)
                  /           \ 
                 V             V
        Image''(B,24,24,32)     Output Value
                |
    1x1-Conv    |
                V
        Logit Maps(B,24,24,6)
                |
    Pos-Masking |
                V
            Logits (B,16,6)
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

        # Transpose image for channel-last format
        image = image.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # Compute value using the value head
        x = res_block_1(conv_block_0(image))
        value = jnp.squeeze(value_head(x.mean(axis=(2, 3))), axis=-1)

        # Gather logits based on position
        logit_maps = conv_block_1(res_block_2(x))
        row_indices, col_indices = position[..., 0], position[..., 1]  # Shape: (N, 16)

        logits = logit_maps[
            jnp.arange(logit_maps.shape[0])[:, None, None],
            row_indices[..., None],
            col_indices[..., None],
            :
        ][:, :, 0, :]  # Shape: (N, 16, 6)
        return logits, value