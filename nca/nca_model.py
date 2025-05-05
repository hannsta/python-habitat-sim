
import torch
import torch.nn as nn
import torch.nn.functional as F
from nca.constsants import SOIL_TYPES, CHANNELS, NUM_CHANNELS, PLANT_RULES, PLANT_GROUPS, W, H
from nca.suitability import compute_suitability
from nca.generate_map import generate_training_world
import random
import time

class NCA(nn.Module):
    def __init__(self, base_channels, feature_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(base_channels + feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x, species_features):
        B, C, H, W = x.shape
        plant_channels = CHANNELS["plants"]
        updated = x.clone()
        proposals = torch.zeros((B, len(plant_channels), H, W), device=x.device)

        plant_list = [name for name, _ in plant_channels]

        for i, (plant, idx) in enumerate(plant_channels):
            plant_map = x[:, idx:idx+1]
            features = species_features[i].view(1, -1, 1, 1).expand(B, -1, H, W)
            input_tensor = torch.cat([x, features], dim=1)


            kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9.0
            
            delta = F.relu(self.model(input_tensor).squeeze(1))
            delta = torch.clamp(delta, 0, 1.0)

            blurred = F.conv2d(plant_map, kernel, padding=1).squeeze(1)
            blurred = torch.clamp(blurred, 0.0, 1.0)

            habitat_suitability = compute_suitability(x, plant, batch_dim=True)
            spread_rate = species_features[i, 1]

            presence = plant_map.squeeze(1)
            is_existing = (presence > 0.1).float()
            is_new = 1.0 - is_existing

            spread_mask = (blurred > 0.05).float() * is_new * spread_rate
            maturation_mask = is_existing

            spread_update = delta * spread_mask * habitat_suitability
            maturation_update = delta * maturation_mask * habitat_suitability

            growth_proposal = torch.clamp(spread_update, 0, 0.1) + maturation_update
            proposals[:, i] = torch.clamp(presence + growth_proposal, 0, 1)


        GROUP_COUNT = 3
        GROUP_OFFSET = -3

        for group_id in range(GROUP_COUNT):
            group_indices = []
            for i, plant_name in enumerate(plant_list):
                group_vector = species_features[i, GROUP_OFFSET:]
                if group_vector[group_id] == 1.0:
                    group_indices.append(i)

            if not group_indices:
                continue

            group_stack = proposals[:, group_indices]
            noise = 0.2 * torch.randn_like(group_stack)
            group_stack = group_stack + noise * (group_stack > 0).float()

            group_competition = species_features[group_indices, 2].view(1, -1, 1, 1)
            weighted_stack = group_stack * group_competition
            winner_strengths, winner_indices = weighted_stack.max(dim=1)

            for local_idx_in_group, global_idx in enumerate(group_indices):
                plant_name = plant_list[global_idx]
                idx = next(idx for name, idx in plant_channels if name == plant_name)
                winner_mask = (winner_indices == local_idx_in_group).float()

                existing = x[:, idx]
                persistence = species_features[global_idx, 3]
                updated[:, idx] = (
                    winner_strengths * winner_mask * (1 - persistence) +
                    existing * persistence
                )

        plant_indices = [idx for _, idx in plant_channels]
        for i in range(x.shape[1]):
            if i not in plant_indices:
                updated[:, i] += x[:, i] * 0.1

        return torch.clamp(updated, 0, 1)

def compute_group_overlap_penalty(grid, species_features):
    """
    Computes overlap penalty within each plant group (grass, shrub, tree)
    based on current active species and their group one-hot encoding.
    """
    penalties = []

    group_count = species_features.shape[1]  # e.g., 17 total features
    group_offset = -3                        # assuming last 3 are group one-hot
    plant_list = [name for name, _ in CHANNELS["plants"]]

    for group_id in range(3):  # 0: grass, 1: shrub, 2: tree
        group_indices = []

        for i, (plant, idx) in enumerate(CHANNELS["plants"]):
            group_vector = species_features[i, group_offset:]
            if group_vector[group_id] == 1.0:
                group_indices.append(idx)

        if not group_indices:
            continue

        group_maps = [grid[:, idx] for idx in group_indices]  # each is [B, H, W]
        group_stack = torch.stack(group_maps)                 # [S, B, H, W]
        group_sum = group_stack.sum(0)                        # [B, H, W]

        overlap = F.relu(group_sum - 1)
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

def compute_loss(grid,species_features, epoch, coverage_deltas):
    if torch.isnan(grid).any():
        print('NaN detected in grid before loss computation.')

    total_penalty = 0
    total_coverage_loss = 0
    competition_penalty = compute_group_overlap_penalty(grid, species_features)
    coverage_reward = 0
    average_coverage_reward = 0
    count =0
    for plant, idx in CHANNELS["plants"]:
        plant_map = grid[:, idx]

        # Suitability-based penalty (already uses binary flags)
        penalty = get_penalty(grid, plant_map, plant)
        total_penalty += penalty

        # Encourage coverage (each plant should reach some % of map)
        coverage = plant_map.mean()
        coverage_target = 0.1
        coverage_loss = (coverage_target - coverage).clamp(min=0).pow(2)
        total_coverage_loss += coverage_loss
        nonzero_mask = (plant_map > 0)
        if nonzero_mask.any():
            avg_coverage = plant_map[nonzero_mask].mean()
        else:
            avg_coverage = torch.tensor(0.0, device=plant_map.device)

        # Incentivize growth
        coverage_reward += coverage
        average_coverage_reward += avg_coverage
        count+=1
    competition_scale = 0.5
    average_coverage_reward = average_coverage_reward/max(count, 1)
    # target = 0.1  # or 0.2 depending on goal
    # reward_term = ((coverage_reward - target).clamp(min=0) ** 2)
    # loss = total_penalty + competition_penalty + total_coverage_loss + 5 * reward_term
    
    clean_deltas = [d for d in coverage_deltas if not torch.isnan(torch.tensor(d)) and not torch.isinf(torch.tensor(d))]
    growth_reward = torch.tensor([max(0.0, d) for d in clean_deltas], device=grid.device).sum()
    #reward_weight = 10.0 * (0.95 ** epoch)  # optional decay
    loss = total_penalty + competition_scale * competition_penalty - coverage_reward * 3 - average_coverage_reward * 5 - growth_reward * 2
    return loss - 10 * growth_reward

def update_shade(grid):

    # with torch.no_grad():
    #     tree_species = PLANT_GROUPS["tree"]
    #     tree_maps = [grid[:, CHANNELS["plants"][p]] for p in tree_species]

    #     # Sum tree presence across all species
    #     combined_trees = torch.stack(tree_maps).sum(0)  # shape: [B, H, W]

    #     # Only proceed if there's any actual tree presence
    #     if combined_trees.max() == 0:
    #         # No trees → clear shade
    #         shade = torch.zeros_like(combined_trees)
    #     else:
    #         # 3x3 neighborhood kernel
    #         combined_trees = combined_trees.unsqueeze(1)  # [B, 1, H, W]
    #         kernel = torch.ones((1, 1, 3, 3), device=grid.device)
    #         neighbor_sum = F.conv2d(combined_trees, kernel, padding=1)

    #         # Normalize: max neighborhood value is 9 if all 1s
    #         shade = torch.clamp(neighbor_sum / 9.0, 0.0, 1.0).squeeze(1)  # [B, H, W]
    #     shade = torch.zeros_like(combined_trees)
    #     grid = grid.clone()
    #     grid[:, CHANNELS["shade"]] = shade
    return grid




def rollout_training_step(model, grid, species_features=None, steps=10, introduce_midway=True):
    elevation_static = grid[:, CHANNELS["elevation"]].clone()
    shade_static = grid[:, CHANNELS["shade"]].clone()
    soil_static = {idx: grid[:, idx].clone() for idx in CHANNELS["soil"].values()}
    if (species_features == None):
        species_features = get_species_features_tensor(device=grid.device)
    previous_coverage = 0.0
    coverage_deltas = []
    for step in range(steps):
        #grid = update_shade(grid)
        grid = model(grid, species_features)
        # Optional intervention: introduce new species midway
        if introduce_midway and step == steps // 2:
            #plant_list = list(CHANNELS["plants"].keys())
            #plant = random.choice(plant_list)
            row, col = torch.randint(0, H, (1,)).item(), torch.randint(0, W, (1,)).item()


            plant, idx = random.choice(CHANNELS["plants"])
            grid[0, idx, row, col] = 1.0
            #grid[0, CHANNELS["plants"][plant], row, col] = 1.0

        # Restore static channels
        grid[:, CHANNELS["shade"]] = shade_static
        grid[:, CHANNELS["elevation"]] = elevation_static
        for idx, value in soil_static.items():
            grid[:, idx] = value
        plant_indices = [idx for _, idx in CHANNELS["plants"]]

        current_coverage = grid[:, plant_indices].sum() / (grid.shape[0] * grid.shape[2] * grid.shape[3])

        # Track growth since last step
        delta = current_coverage - previous_coverage
        coverage_deltas.append(delta)
        previous_coverage = current_coverage

    return grid, species_features, coverage_deltas

def get_species_features_tensor(device="cuda", species_list=None):
    vectors = []

    if species_list is None:
        species_list = [name for name, _ in CHANNELS["plants"]]
    for plant in species_list:
        rule = PLANT_RULES[plant]

        core_features = torch.tensor([
            rule["moisture_tolerance"],
            rule["spread_rate"],
            rule["root_competition"],
            rule["persistence"]
        ] + rule["soil_types"] + rule["elevation"] + rule["shade"], device=device)

        group_vector = torch.tensor(rule["group"], dtype=torch.float32, device=device)

        vec = torch.cat([core_features, group_vector])
        vectors.append(vec)

    return torch.stack(vectors)

import random
from nca.constsants import PLANT_RULES, CHANNELS

def randomize_species_order(base_channel=6):
    plant_order = list(PLANT_RULES.keys())
    random.shuffle(plant_order)

    CHANNELS["plants"] = [(plant_name, base_channel + i) for i, plant_name in enumerate(plant_order)]

    return plant_order


def build_channel_mapping_from_species_list(species_list, base_channel=6):
    from nca.constsants import CHANNELS as global_channels
    global_channels["plants"] = [(plant_name, base_channel + i) for i, plant_name in enumerate(species_list)]

    return global_channels["plants"]



def log_channel_sums(grid: torch.Tensor):
    assert grid.ndim == 4, "Expected shape [1, channels, height, width]"
    channels = grid.shape[1]
    for c in range(channels):
        channel_sum = grid[0, c].sum().item()
        print(f"Channel {c} sum: {channel_sum}")