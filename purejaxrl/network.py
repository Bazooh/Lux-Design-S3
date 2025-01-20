import jax.numpy as jnp
import jax
import flax.linen as nn
from typing import Sequence
from flax.linen.initializers import constant, orthogonal

class Flatten(nn.Module):
    def __call__(self, x):
        return x.reshape(x.shape[0], -1)
    
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
        y = nn.Dense(self.channel // self.reduction, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(y)
        y = nn.relu(y)
        y = nn.Dense(self.channel, self.channel // self.reduction, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(y)
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
            nn.Conv(
                features = self.out_channel, 
                kernel_size=self.kernel_size, 
                strides=1, 
                padding=self.padding, 
                kernel_init=orthogonal(jnp.sqrt(2)),
            ),
            nn.leaky_relu,
        ])# conv block
        se_layer = SELayer(self.out_channel, reduction=16)
        out = x + se_layer(conv_block(x))
        out = nn.leaky_relu(out)
        return out

class Pix2Pix_AC(nn.Module):
    """

                    Image (B,C,24,24)    
                        |                
                        | Transpose      
                        V              
                    Image (B,24,24,C)     Vector (B,V)
                        |                /
                        | Concat        / FC + Expand  
                        V              V  
                    Image (B,24,24,C')                      
                        |
                        | 1x1-Conv
                        V
                    Image (B,24,24,32)                      
                        |
                        | ResBlock 1
                        V
                    Image'(B,24,24,32)
                    /       \  
        ResBlock2  /         \  Value Head
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
        conv1_1 = nn.Sequential([
            nn.Conv(64, kernel_size=1, strides=1, padding=0, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
        ], name='conv1_1') # conv 1x1 block 
        conv1_2 = nn.Sequential([
            nn.Conv(self.action_dim, kernel_size=1, strides=1, padding=0, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
        ], name='conv1_2') # conv 1x1 block 
        fc_vec = nn.Sequential([
            nn.Dense(16, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.leaky_relu,
        ])
        res_blocks_1 = nn.Sequential([ResidualBlock(in_channel=64, out_channel=64, kernel_size=3, padding=1) for _ in range(4)])
        res_blocks_2 = nn.Sequential([ResidualBlock(in_channel=64, out_channel=64, kernel_size=3, padding=1) for _ in range(2)])
        value_head = nn.Sequential([
            nn.Conv(16, kernel_size=1, strides=1, padding=0, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.leaky_relu,
            nn.Conv(1, kernel_size=1, strides=1, padding=0, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.leaky_relu,
            Flatten(),
            nn.leaky_relu,
            nn.Dense(16, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.leaky_relu,
            nn.Dense(1, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
        ], name="value_head")

        # Transpose image for channel-last format
        image = image.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        B, H, W, C = image.shape

        # Process the vector
        vector = fc_vec(vector) # (N, V) -> (N, 16)
        vector = vector.reshape((B, 1, 1, -1))  # (N, 16) -> (N, 1, 1, 16)
        vector = jnp.tile(vector, (1, H, W, 1))  # (N, 1, 1, 16) -> (N, H, W, 16)

        # Concatenate
        x = jnp.concatenate((image, vector), axis=-1)  # (N, H, W, C + 16)
        
        # 1x1 Conv
        x = conv1_1(x)  # (N, H, W, C + 16) -> (N, H, W, 32) 
        
        # Resblock1
        x = res_blocks_1(x)
        
        # Compute value using the value head
        value = jnp.squeeze(value_head(x), axis=-1)

        # Resblock2
        x = res_blocks_2(x)

        # Compute logit maps: 1x1 Conv
        logits_maps = conv1_2(x)

        # Gather logits based on position
        def gather_logits(logits_map, pos):
            # logits_map: Shape (24, 24, 6)
            # pos: Shape (16, 2)
            return logits_map[pos[:, 0], pos[:, 1], :]  # Shape (16, 6)
        
        logits_gathered = jax.vmap(gather_logits)(logits_maps, position) # Shape: (N, 16, 6)
        return logits_gathered, value