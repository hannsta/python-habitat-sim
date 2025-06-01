import torch
import torch.nn as nn
import torch.nn.functional as F
from nca.constsants import NUM_CHANNELS

class PolicyNet(nn.Module):
    def __init__(self, num_species, h, w, species_feature_dim=18, attn_temp=0.5):
        super().__init__()
        self.attn_temp = attn_temp  # softmax temperature

        # Convolutional backbone with 2x downsampling
        self.conv1 = nn.Conv2d(NUM_CHANNELS + 1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Attention projection
        self.grid_proj = nn.Linear(130, 64)  # adjusted for 128 + 2 (positional channels)

        self.species_proj = nn.Sequential(
            nn.Linear(species_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.species_query_proj = nn.Linear(64, 64)

        self.attn_score_head = nn.Sequential(
            nn.Linear(130, 64),  # adjusted input size to match attended_env features
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Upsample to high-res logits (from H/4 → H/2 → H)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, species_features):
        B = x.size(0)
        device = x.device

        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, H/4, W/4]

        # Positional encoding
        _, _, H, W = x.shape
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        pos = torch.stack([(yy.float() / H - 0.5), (xx.float() / W - 0.5)], dim=-1).to(device)  # [H, W, 2]
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
        pos = pos.permute(0, 3, 1, 2)  # [B, 2, H, W]
        x_with_pos = torch.cat([x, pos], dim=1)  # [B, 130, H, W]

        # Flatten and project
        x_flat = x_with_pos.view(B, x_with_pos.size(1), -1).permute(0, 2, 1)  # [B, H*W, 130]
        env_proj = self.grid_proj(x_flat)  # [B, H*W, 64]

        # Species encoding
        # species_embed = self.species_proj(species_features)  # [S, 64]
        # query_embed = self.species_query_proj(species_embed)  # [S, 64]

        # env_proj = F.normalize(env_proj, dim=-1)
        # query_embed = F.normalize(query_embed, dim=-1)

        # S = species_embed.size(0)
        # query_exp = query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, S, 64]

        if species_features.dim() == 2:
            # In reinforcement mode: shape [S, F]
            species_embed = self.species_proj(species_features)  # [S, 64]
            query_embed = self.species_query_proj(species_embed)  # [S, 64]
            query_embed = F.normalize(query_embed, dim=-1)
            query_exp = query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, S, 64]
        else:
            # In supervised training: shape [B, S, F]
            species_embed = self.species_proj(species_features)  # [B, S, 64]
            query_embed = self.species_query_proj(species_embed)  # [B, S, 64]
            query_exp = F.normalize(query_embed, dim=-1)




        similarity = torch.matmul(query_exp, env_proj.transpose(1, 2))  # [B, S, H*W]
        attn_weights = F.softmax(similarity / self.attn_temp, dim=-1)  # sharper attention

        attended_env = torch.bmm(attn_weights, x_flat)  # [B, S, 130]
        species_logits = self.attn_score_head(attended_env).squeeze(-1)  # [B, S]

        # High-resolution spatial placement
        location_logits = self.upsample(x).reshape(B, -1)

        return species_logits, location_logits
