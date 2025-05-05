import torch.nn as nn
import torch
import torch.nn.functional as F

from nca.constsants import NUM_CHANNELS

class PolicyNet(nn.Module):
    def __init__(self, num_species, h, w, species_feature_dim=18):
        super().__init__()

        self.conv1 = nn.Conv2d(NUM_CHANNELS + 1, 32, 3, padding=1)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.AvgPool2d(2)

        self.location_head = nn.Conv2d(32, 1, 1)

        self.avgpool = nn.AvgPool2d(kernel_size=(h // 4, w // 4))
        self.env_proj = nn.Linear(32, 64)
        self.species_proj = nn.Linear(species_feature_dim, 64)

        self.classifier = nn.Linear(64, 1)

    def forward(self, x, species_features):
        x1 = F.relu(self.conv1(x))
        x2 = self.pool1(x1)
        x3 = F.relu(self.conv2(x2))
        x4 = self.pool2(x3)
        x4 = torch.clamp(x4, 0, 6)

        env_feat = self.avgpool(x4).view(x.size(0), -1)             # [B, 32]
        env_embed = self.env_proj(env_feat)                         # [B, 64]
        species_embed = self.species_proj(species_features)        # [S, 64]

        # Cross env and species: [B, 64] @ [S, 64].T → [B, S]
        species_logits = torch.matmul(env_embed, species_embed.T)

        location_logits = self.location_head(x4).reshape(x.size(0), -1)

        return species_logits, location_logits



import onnxruntime as ort
import torch
import numpy as np

class PolicyNetONNX:
    def __init__(self, model_path, device='cuda'):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.device = device

    def __call__(self, grid):
        if grid.is_cuda:
            input_numpy = grid.detach().cpu().numpy()
        else:
            input_numpy = grid.detach().numpy()

        outputs = self.session.run(None, {"input": input_numpy})

        species_logits = torch.from_numpy(outputs[0]).to(self.device)
        location_logits = torch.from_numpy(outputs[1]).to(self.device)

        # Match your original PyTorch model behavior (optional)
        species_logits = species_logits + 0.1 * torch.randn_like(species_logits)
        location_logits = location_logits + 0.1 * torch.randn_like(location_logits)

        return species_logits, location_logits