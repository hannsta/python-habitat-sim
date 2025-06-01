import random
import torch
import numpy as np
from nca.generate_map import generate_training_world
from nca.nca_model import build_channel_mapping_from_species_list, get_species_features_tensor, rollout_training_step, NCA
from nca.suitability import compute_suitability
from nca.constsants import CHANNELS, NUM_CHANNELS, PLANT_RULES, H, W
import torch
import numpy as np
from nca.generate_map import generate_training_world
from nca.nca_model import get_species_features_tensor, rollout_training_step, NCA
from nca.suitability import compute_suitability
from nca.constsants import CHANNELS, PLANT_RULES, H, W

def build_quadrant_mask(player_number):
    mask = torch.zeros((1, H, W), device="cuda")
    if player_number == 1:
        mask[0, :H//2, :W//2] = 1.0
    else:
        mask[0, H//2:, W//2:] = 1.0
    return mask

def generate_supervised_dataset(num_samples=50, steps_per_sample=10, trials_per_species=30):
    dataset = []
    nca_model = NCA(base_channels=16, feature_dim=18).to("cuda")
    nca_model.eval()


    for sample_idx in range(num_samples):
        grid, _ = generate_training_world(H, W, seed_plants=False, seed_smart=False)
        grid = grid.to("cuda")

        player_number = np.random.choice([1, 2])
        species_list = random.sample(list(PLANT_RULES.keys()), k=10)

        if player_number == 1:
            legal_species_ids = list(range(5))  # species 0-4
        else:
            legal_species_ids = list(range(5, 10))  # species 5-9

        quadrant_mask = build_quadrant_mask(player_number)  # shape [1, H, W]
        augmented_grid = torch.cat([grid[0], quadrant_mask[0].unsqueeze(0)], dim=0).unsqueeze(0)  # [1, C+1, H, W]

        # Build full species_features with ownership flags
        base_features = get_species_features_tensor(device="cuda", species_list=species_list)
        ownership_flags = torch.tensor(
            [1.0 if (player_number == 1 and i < 5) or (player_number == 2 and i >= 5) else 0.0 for i in range(10)],
            device="cuda"
        ).unsqueeze(1)
        species_features = torch.cat([base_features, ownership_flags], dim=1)  # shape [S, F+1]
        build_channel_mapping_from_species_list(species_list) 

        best_score = -float("inf")
        best_species_id = None
        best_location = None

        valid_placements = []

        for species_id in legal_species_ids:
            species_name = species_list[species_id]
            suitability = compute_suitability(grid[0], species_name, batch_dim=False)

            for _ in range(trials_per_species):
                row = torch.randint(0, H, (1,)).item()
                col = torch.randint(0, W, (1,)).item()

                if suitability[row, col] < 0.2 or quadrant_mask[0, row, col] < 0.5:
                    continue

                test_grid = grid.clone()
                channel = CHANNELS["plants"][species_id][1]
                test_grid[0, channel, row, col] = 1.0

                with torch.no_grad():
                    result_grid, _, coverage_deltas = rollout_training_step(
                        model=nca_model,
                        grid=test_grid,
                        species_features=species_features,
                        steps=steps_per_sample,
                        introduce_midway=False
                    )

                growth_score = sum(max(0.0, d) for d in coverage_deltas)

                if growth_score > 0:
                    valid_placements.append({
                        "species_id": species_id,
                        "species_name": species_name,
                        "location": (row, col),
                        "score": growth_score
                    })
        # Filter to only good enough
        valid_placements = [p for p in valid_placements if p["score"] > 0]

        if valid_placements:
            # Prefer top 3, pick randomly among them
            top_choices = sorted(valid_placements, key=lambda x: -x["score"])[:3]
            chosen = random.choice(top_choices)

            dataset.append({
                "grid": augmented_grid.cpu(),
                "species_features": species_features.cpu(),
                "target_species_id": chosen["species_id"],
                "target_location": chosen["location"],
                "player_number": player_number
            })
        else:
            print(f"[Skipped] No valid placements above growth=0 for species {species_name}, map variety={_}")

        print(f"Sample {sample_idx+1}/{num_samples} complete.")

    return dataset


from torch.utils.data import Dataset, DataLoader
import torch

class PlantPolicyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        grid = sample["grid"]  # [1, C, H, W]
        species_features = sample["species_features"]  # [S, F]
        target_species_id = sample["target_species_id"]
        row, col = sample["target_location"]
        flat_target_location = row * W + col

        return {
            "grid": grid,
            "species_features": species_features,
            "target_species_id": torch.tensor(target_species_id, dtype=torch.long),
            "target_location": torch.tensor(flat_target_location, dtype=torch.long)
        }
