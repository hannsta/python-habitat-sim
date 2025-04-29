
import torch
import torch.nn as nn
import torch.nn.functional as F
from game.constsants import SOIL_TYPES, PLANT_TYPES, CHANNELS, NUM_CHANNELS, PLANT_RULES, PLANT_GROUPS, W, H
from game.suitability import compute_suitability
from game.generate_map import generate_training_world
import random

class NCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, channels, kernel_size=1)
        )
    def forward(self, x):
        x = x.clone()
        update = self.model(x) * 0.1  # scale down updates to be safe

        plant_channels = CHANNELS["plants"]
        updated = x

        # --- Step 1: Build proposal tensor ---
        proposals = torch.zeros((x.size(0), len(plant_channels), x.size(2), x.size(3)), device=x.device)

        plant_list = list(plant_channels.keys())
        channel_list = list(plant_channels.values())

        # First: Each plant proposes its new map
        for i, plant in enumerate(plant_list):
            idx = plant_channels[plant]

            plant_map = x[:, idx:idx+1]
            kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9.0
            blurred = F.conv2d(plant_map, kernel, padding=1)
            blurred = torch.clamp(blurred, 0.0, 1.0)

            spread_mask = (blurred > 0.1).float()

            habitat_suitability = compute_suitability(x, plant, batch_dim=True)
            spread_weight = spread_mask.squeeze(1) + 0.2  # Open areas have +1.2, occupied +0.2

            #growth_proposal = (x[:, idx] + update[:, idx]) * habitat_suitability * spread_weight

            # growth_proposal = (x[:, idx] + update[:, idx]) * habitat_suitability
            growth_proposal = update[:, idx] * spread_mask.squeeze(1) * habitat_suitability
            proposals[:, i] = x[:, idx] + growth_proposal  # proposed next state

        # --- Step 2: Contest inside each group ---
        for group_name, species in PLANT_GROUPS.items():
            group_indices = [plant_list.index(p) for p in species]

            group_stack = proposals[:, group_indices]  # [B, num_species_in_group, H, W]

            # Inject small noise ONLY where proposals exist
            noise = 0.2 * torch.randn_like(group_stack)
            group_stack = group_stack + noise * (group_stack > 0).float()

            # Then:
            winner_strengths, winner_indices = group_stack.max(dim=1)

            for i, plant_name in enumerate(species):
                idx = plant_channels[plant_name]
                local_idx = plant_list.index(plant_name)

                winner_mask = (winner_indices == i).float()

                # Update grid: only plant that wins at each pixel keeps its value
                updated[:, idx] = winner_strengths * winner_mask

        # --- Step 3: Update other channels (soil, elevation, shade) normally ---
        for i in range(x.shape[1]):
            if i not in plant_channels.values():
                updated[:, i] += update[:, i]

        # Clamp final grid
        return torch.clamp(updated, 0, 1)


def compute_group_overlap_penalty(grid, groups):
    penalties = []
    for group in groups.values():
        group_maps = [grid[:, CHANNELS["plants"][p]] for p in group]
        group_stack = torch.stack(group_maps)  # [num_species, B, H, W]
        group_sum = group_stack.sum(0)         # [B, H, W]
        
        overlap = F.relu(group_sum - 1)

        # Scale penalty properly
        penalty = overlap.sum() / (group_sum.numel() + 1e-8)
        penalties.append(penalty)
    
    if penalties:
        return torch.stack(penalties).mean()
    else:
        return torch.tensor(0.0, device=grid.device)


# --- Helper: Constraint Mapping ---
def get_penalty(grid, plant_map, plant_name):

    suitability = compute_suitability(grid, plant_name)
    
    penalty = (1 - suitability) * plant_map
    return penalty.mean()

def compute_loss(grid, epoch):
    elevation = grid[:, CHANNELS["elevation"]]
    shade = grid[:, CHANNELS["shade"]]
    if torch.isnan(grid).any():
        print('NaN detected in grid before loss computation.')
    total_penalty = 0
    total_coverage_loss = 0
    competition_penalty = compute_group_overlap_penalty(grid, PLANT_GROUPS)
    coverage_reward = 0
    for plant, idx in CHANNELS["plants"].items():
        plant_map = grid[:, idx]
        rule = PLANT_RULES[plant]
        soil_mask = grid[:, CHANNELS["soil"][rule["soil"]]]

        penalty = get_penalty(grid, plant_map, plant)
        coverage = plant_map.mean()

        # Encourage coverage > ε (e.g., 1% of map)
        coverage_target = 0.1
        coverage_loss = (coverage_target - coverage).clamp(min=0).pow(2)
        coverage_reward += coverage

        total_penalty += penalty
        total_coverage_loss += coverage_loss
    competition_scale = .5
    return  1.5 * total_penalty + competition_scale * competition_penalty - 1 * coverage_reward

# --- Shade Calculation ---
def update_shade(grid):
    with torch.no_grad():
        tall = torch.stack([
            grid[:, CHANNELS["plants"]["tree_0"]],
            grid[:, CHANNELS["plants"]["tree_1"]],
        ]).sum(0, keepdim=True)
        kernel = torch.ones((1, 1, 3, 3), device=grid.device)
        shade = F.conv2d(tall, kernel, padding=1) / 8.0
        grid[:, CHANNELS["shade"]] = torch.clamp(shade.squeeze(1), 0, 1)

def rollout_training_step(model, grid, steps=10, introduce_midway=True):
    elevation_static = grid[:, CHANNELS["elevation"]].clone()
    soil_static = {idx: grid[:, idx].clone() for idx in CHANNELS["soil"].values()}

    for step in range(steps):
        update_shade(grid)
        grid = model(grid)
        
        # Optional intervention: introduce new species midway
        if introduce_midway and step == steps // 2:
            plant_list = list(CHANNELS["plants"].keys())
            plant = random.choice(plant_list)
            row, col = torch.randint(0, H, (1,)).item(), torch.randint(0, W, (1,)).item()
            grid[0, CHANNELS["plants"][plant], row, col] = 1.0


        # Restore static channels
        grid[:, CHANNELS["elevation"]] = elevation_static
        for idx, value in soil_static.items():
            grid[:, idx] = value

    return grid