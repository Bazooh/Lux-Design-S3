import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Conv1x1(nn.Module):
    """
    Convolution 1x1
    """
    def __init__(self, channels, with_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.with_relu = with_relu
    
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x) if self.with_relu else x

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    """
    def __init__(self, n_channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(n_channels, n_channels // reduction)
        self.fc2 = nn.Linear(n_channels // reduction, n_channels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        vector = self.global_avg_pool(x).view(B, C)  # (B, C)
        squeezed_vector = F.relu(self.fc1(vector))  # (B, C")
        unsqueezed_vector = torch.sigmoid(self.fc2(squeezed_vector)).view(B, C, 1, 1)  # (B, C) -> (B, C, 1, 1)
        return x * unsqueezed_vector

class ResidualBlock(nn.Module):
    """
    Residual Block
    """
    def __init__(self, n_channels, kernel_size, strides, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=strides, padding=padding, bias=True)
        self.se_block = SEBlock(n_channels)
    
    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = self.se_block(out)
        return F.leaky_relu(out + residual)
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class Pix2Pix_AC(nn.Module):
    def __init__(
        self, 
        action_dim=6, 
        n_resblocks=6, 
        n_channels=64, 
        embedding_time=10, 
        dim_time=55,
        dim_vector=18,
        dim_image=36,
        normalize_logits=True, 
        normalize_value=True, 
        action_masking=True, 
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_resblocks = n_resblocks
        self.n_channels = n_channels
        self.embedding_time = embedding_time
        self.normalize_logits = normalize_logits
        self.normalize_value = normalize_value
        self.action_masking = action_masking

        self.fc_time = nn.Sequential(nn.Linear(dim_time, embedding_time), nn.LeakyReLU())
        self.conv1x1_time = nn.Conv2d(embedding_time, embedding_time, kernel_size=1)
        self.conv1x1_vec = nn.Conv2d(dim_vector, dim_vector, kernel_size=1)
        self.conv1x1_time_vec = nn.Conv2d(embedding_time + dim_vector, embedding_time + dim_vector, kernel_size=1)
        self.conv1x1_input = nn.Conv2d(dim_image + embedding_time + dim_vector, n_channels, kernel_size=1)
        self.conv1x1_logits = nn.Conv2d(n_channels, action_dim, kernel_size=1)
        self.conv1x1_points = nn.Conv2d(n_channels, 1, kernel_size=1)
        self.spectral_norm = nn.utils.spectral_norm(nn.Conv2d(n_channels, n_channels, kernel_size=1))

        self.res_blocks = nn.Sequential(*[ResidualBlock(n_channels, kernel_size=5, padding=2, strides=1) for _ in range(n_resblocks)])
        self.value_head = nn.Linear(n_channels, 1)

    def forward(self, image, vector, time, position, mask_awake, action_mask, train=False):
        B, C, H, W = image.shape
        B, V = vector.shape
        B, T = time.shape

        time_embedded = self.fc_time(time)
        time_expanded = time_embedded.view(B, 1, 1, -1).expand(-1, H, W, -1).permute(0, 3, 1, 2)
        time_expanded = self.conv1x1_time(time_expanded)

        vector_expanded = vector.view(B, 1, 1, -1).expand(-1, H, W, -1).permute(0, 3, 1, 2)
        vector_expanded = self.conv1x1_vec(vector_expanded)

        vector = torch.cat((time_expanded, vector_expanded), dim=1)
        vector = self.conv1x1_time_vec(vector)

        image = image.permute(0, 2, 3, 1).contiguous()
        x = torch.cat((image, vector), dim=1)
        x = self.conv1x1_input(x)
        x = self.res_blocks(x)
        x_normalized = self.spectral_norm(x) if train else x

        average = torch.mean(x_normalized if self.normalize_value else x, dim=(2, 3))
        value = self.value_head(average).squeeze(-1)

        logits_maps = self.conv1x1_logits(x_normalized if self.normalize_logits else x)
        points = torch.sigmoid(self.conv1x1_points(x_normalized if self.normalize_logits else x))

        logits_gathered = torch.stack([logits_maps[b, :, pos[:, 0], pos[:, 1]] for b, pos in enumerate(position)])
        logits = logits_gathered * mask_awake.unsqueeze(-1)
        logits_masked = torch.where(action_mask, logits, torch.tensor(1e-9))
        return logits_masked if self.action_masking else logits, value, logits_maps, points
