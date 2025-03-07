import jax.numpy as jnp
import jax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

class Flatten(nn.Module):
    def __call__(self, x):
        return x.reshape(x.shape[0], -1)

class Conv1x1(nn.Module): 
    """
    Convolution 1x1
    """
    channels: int
    with_relu: bool = True
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, 
                    kernel_size=1, 
                    strides=1, 
                    padding=0, 
                    kernel_init=orthogonal(jnp.sqrt(2)), 
                    bias_init=constant(0.0), 
                    use_bias = True)(x)
        return nn.leaky_relu(x) if self.with_relu else x

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    """
    n_channels: int 
    reduction: int = 16 # C" = C / reduction

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        # Squeeze: Global Average Pooling
        vector = jnp.mean(x, axis=(1, 2), keepdims=False)  # (B, H, W, C) -> (B, C)
        # Excitation: Fully Connected layers with reduction
        squeezed_vector = nn.Dense(self.n_channels // self.reduction, 
                                   kernel_init=orthogonal(jnp.sqrt(2)), 
                                   bias_init=constant(0.0))(vector) # (B, C) -> (B, C")
        squeezed_vector = nn.relu(squeezed_vector)
        unsqueezed_vector = nn.Dense(self.n_channels, 
                                    self.n_channels // self.reduction, 
                                    kernel_init=orthogonal(jnp.sqrt(2)), 
                                    bias_init=constant(0.0))(squeezed_vector)  # (B, C") -> (B, C)  
        # Go through sigmoid
        out = nn.sigmoid(unsqueezed_vector)  # (B, C)
        out_expanded = jnp.tile(out.reshape((B, 1, 1, -1)), (1, H, W, 1))  # (B, C) -> (B, H, W, C)
        # Rescale original image
        return x * out_expanded

class ResidualBlock(nn.Module):
    """
    Residual Block:
                  x
               /    \                  
              /      \  
             /        \ 
            V          \ 
    conv kxk + Lrelu    \ 
        |                |
        v                |  shortcut
    conv kxk + Lrelu     |
        |                |
        v               /
    SE-BLOCK           /
            \         / 
             \  add  /  
              V     V
                out
                |   Lrelu
                v
                out
    """
    n_channels: int
    kernel_size: int 
    strides: int
    padding: int 
    
    @nn.compact
    def __call__(self, x):
        conv_blocks = nn.Sequential([
            nn.Conv(
                features = self.n_channels, 
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding=self.padding, 
                kernel_init=orthogonal(jnp.sqrt(2)),
                use_bias = True
            ),
            nn.leaky_relu,
            nn.Conv(
                features = self.n_channels, 
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding=self.padding, 
                kernel_init=orthogonal(jnp.sqrt(2)),
                use_bias = True
            ),
            nn.leaky_relu,
        ])
        se_layer = SEBlock(self.n_channels)

        out = nn.leaky_relu(x + se_layer(conv_blocks(x)))
        return out

class Pix2Pix_AC(nn.Module):
    """
                                        Time OHE (B,55)                 Global game features (B, 18)
                                            |                                          |
                                            |    FC                                    |
                                            v                                          |
                                        Embedded Time (B,10)                           | expand + 1x1 Conv 
                                            |                                          | 
                                            | expand + 1x1 Conv                        |  
                                            v                                          v
                                       Expanded (B, 24, 24, 10)      Expanded (B, 24, 24, 18)
                                                    \                  /
                                                     \     Concat     /              
                                                      \              /                        
                                                       v            v
                Map features (B,8,24,24)           Vector (B,24,24,28)
                    |                                  |
                    |   Transpose                      |  1x1 conv 
                    |                                  v
                Map features (B,24,24,8)             Vector (B,24,24,28)
                    |                                 / 
                    |   Concat         ______________/ 
                    |                 /                 
                    V                V  
                Image (B,24,24,36)                      
                    |
                    |   1x1-Conv
                    |
                    V
                Input (B,24,24,64)                      
                    |
                    |   ResBlocks
                    | 
                    V
                Representation (B,24,24,64)
                    |
                    |  Spectral Norm
                    | 
                    V
                Representation Normalized (B,24,24,64)
                /                   |                   \  
1x1-Conv       /                    |                    \  AvgPool2D
              /                     |                     \ 
             V                      v                      V
    Logit Maps(B,24,24,6)      Point Pred (B,24,24)   Avg (B, 64)
            |                                               |
            |                                               |
            |                                               |                               
Pos-Masking |                                               |    Value Head
            |                                               |
            V                                               V
        Logits (B,16,6)                                 Value (B, 1)
    """
    
    action_dim: int = 6
    n_resblocks: int = 6
    n_channels: int = 64
    embedding_time: int = 10
    normalize_logits: bool = True
    normalize_value: bool = True
    action_masking: bool = True

    @nn.compact
    def __call__(self, image, vector, time, position,  mask_awake, action_mask, train = False):
        
        B, C, H, W = image.shape
        B, V = vector.shape
        B, T = time.shape
        
        fc_time = nn.Sequential([nn.Dense(self.embedding_time, kernel_init=orthogonal(jnp.sqrt(2))), nn.leaky_relu], name="fc_time")
        fc_vec = nn.Sequential([nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2))), nn.leaky_relu], name="fc_vec")
        fc_time_vec = nn.Sequential([nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2))), nn.leaky_relu], name="fc_time_vec")
        conv1x1_input = Conv1x1(channels=self.n_channels, name="conv1x1_input") # conv 1x1 block
        conv1x1_logits = Conv1x1(channels=self.action_dim, with_relu = False, name="conv1x1_logits") # conv 1x1 block  
        conv1x1_points = Conv1x1(channels=1, with_relu = False, name="conv1x1_points") # conv 1x1 block  
        spectral_norm = nn.SpectralNorm(Conv1x1(channels=self.n_channels, name="spectral_norm"))

        res_blocks = nn.Sequential([ResidualBlock(n_channels=self.n_channels, kernel_size=5, padding=2, strides=1) for _ in range(self.n_resblocks)], name="res_blocks")
        
        value_head = nn.Dense(1, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
        
        ########## PROCESS TIME AND VECTOR ##########
        time_embedded = fc_time(time) # (B, T) -> (B, embedding_time)
        vector_embedded = fc_vec(vector)
        time_and_vector = jnp.concatenate((time_embedded, vector_embedded), axis=-1) 
        time_and_vector_embedded = fc_time_vec(time_and_vector) 
        time_and_vector_expanded = jnp.tile(time_and_vector_embedded.reshape((B, 1, 1, -1)) , (1, H, W, 1)) # (B, V) -> (B, H, W, V)

        ################# COMBINE WITH IMAGE ################
        image = image.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        
        x = jnp.concatenate((image, time_and_vector_expanded), axis=-1)  # (B, H, W, C")
        
        x = conv1x1_input(x) # (B, H, W, n_channels)
        
        x = res_blocks(x) # (B, H, W, n_channels)

        x_normalized = spectral_norm(x, update_stats = train)

        ################# Compute VALUE  ################
        
        average = jax.lax.select(
            self.normalize_value,
            jnp.mean(x_normalized, axis=(1, 2)),
            jnp.mean(x, axis=(1, 2))
        )  # (B, H, W, n_channels) -> (B, n_channels)
        value = value_head(average) # (B, n_channels) -> (B, 1)
        value = jnp.squeeze(value, axis=-1)


        ################# Compute LOGITS  ################
        logits_maps = jax.lax.select(
            self.normalize_logits,
            conv1x1_logits(x_normalized),
            conv1x1_logits(x)
        )
        ################# Compute Points ################
        points = jax.lax.select(
            self.normalize_logits,
            conv1x1_points(x_normalized),
            conv1x1_points(x)
        )
        points = nn.sigmoid(points)

        # Gather logits based on position
        def gather_logits(logits_map, pos):
            # logits_map: Shape (24, 24, 6)
            # pos: Shape (16, 2)
            return logits_map[pos[:, 0], pos[:, 1], :]  # Shape (16, 6)
        
        logits_gathered = jax.vmap(gather_logits)(logits_maps, position) # Shape: (B, 16, 6)
        
        mask_awake_expanded = mask_awake[:, :, None]  # Expand dimensions to (1, 16, 1)
        logits = logits_gathered * mask_awake_expanded 
        logits_masked = jnp.where(action_mask, logits, 1e-9)
        return jax.lax.select(self.action_masking, logits_masked, logits), value, logits_maps, points