import numpy as np
import torch
import scipy.ndimage as ndi

from nca.constsants import CHANNELS, NUM_CHANNELS, PLANT_RULES, PLANT_GROUPS, SOIL_TYPES
from nca.suitability import compute_suitability
from nca.constsants import W, H

def generate_elevation_map(h, w, passes=3, falloff=True, variety_type="default"):
    """Generates elevation maps with different terrain styles."""
    elevation = np.zeros((h, w), dtype=np.float32)

    for _ in range(passes):
        noise = np.random.rand(h, w)
        blurred = ndi.gaussian_filter(noise, sigma=5)
        elevation += blurred

    elevation /= passes
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())

    if falloff:
        y, x = np.ogrid[:h, :w]
        cx, cy = h / 2, w / 2
        d = np.sqrt((x - cx)**2 + (y - cy)**2)

        if variety_type == "default":
            mask = 1 - np.clip(d / (0.9 * max(h, w) / 2), 0, 1)
        elif variety_type == "offcenter":
            # Move the center randomly
            shift_x = np.random.uniform(-0.2, 0.2) * w
            shift_y = np.random.uniform(-0.2, 0.2) * h
            cx_shifted = cx + shift_x
            cy_shifted = cy + shift_y
            d = np.sqrt((x - cx_shifted)**2 + (y - cy_shifted)**2)
            mask = 1 - np.clip(d / (0.8 * max(h, w) / 2), 0, 1)
        elif variety_type == "ridge":
            # Generate a ridge instead of a centered island
            mask = 1 - np.clip(np.abs(x - cx) / (0.5 * w), 0, 1)
        elif variety_type == "basin":
            # Basin-like structure
            mask = np.clip(d / (0.7 * max(h, w) / 2), 0, 1)
        else:
            mask = np.ones_like(elevation)

        elevation *= mask

    elevation = np.clip(elevation, 0, 1)
    return elevation

def compute_slope(elevation):
    """Estimates slope magnitude from elevation."""
    dy, dx = np.gradient(elevation)
    slope = np.sqrt(dx**2 + dy**2)
    return (slope - slope.min()) / (slope.max() - slope.min())

def generate_soil_map(elevation, slope, num_types=4, noise_strength=0.2):
    """Classifies soil types based on slope and noise."""
    noise = np.random.rand(*slope.shape) * noise_strength
    combined = slope + noise
    thresholds = np.linspace(0, 1, num_types + 1)[1:-1]
    soil = np.digitize(combined, thresholds)
    return soil  # values from 0 to num_types - 1

def generate_training_world(H, W, seed_plants = True, total_plant_species = 10, seed_smart = True):
    elevation = elevation = generate_elevation_map(H, W, passes=4, falloff=True, variety_type=np.random.choice(["default", "offcenter", "ridge", "basin"]))
    slope = compute_slope(elevation)
    soil = generate_soil_map(elevation, slope, num_types=4)

    # Convert to tensors
    elevation_tensor = torch.tensor(elevation, dtype=torch.float32)
    soil_tensor = torch.zeros(4, H, W)
    for i in range(4):
        soil_tensor[i] = (torch.tensor(soil) == i).float()


    total_channels = total_plant_species + 6
    # Assemble grid
    grid = torch.zeros(1, total_channels, H, W)
    grid[0, CHANNELS["elevation"]] = elevation_tensor
    for soil_name, soil_idx in CHANNELS["soil"].items():
        grid[0, soil_idx] = soil_tensor[soil_idx]

    grid[0, CHANNELS["shade"]] = torch.zeros(H,W)

    # Static values
    grid[0, CHANNELS["elevation"]] = elevation_tensor
    
    
    if (seed_plants):
        if (seed_smart):
            seed_plants_smart(grid)  # Add this line
        else:
            seed_plants_random(grid, 8)
    return grid
def seed_plants_random(grid, num_seeds_per_plant=3):
    B, C, H, W = grid.shape

    for plant_name, idx in CHANNELS["plants"]:
        for _ in range(num_seeds_per_plant):
            row = torch.randint(0, H, (1,)).item()
            col = torch.randint(0, W, (1,)).item()
            grid[0, idx, row, col] = 1.0

    return grid
def get_plant_channel_index(plant_name):
    for name, idx in CHANNELS["plants"]:
        if name == plant_name:
            return idx
    raise ValueError(f"Plant '{plant_name}' not found in CHANNELS['plants']")
def seed_plants_smart(grid, num_seeds_per_plant=3, min_dist=5):
        elevation = grid[0, CHANNELS["elevation"]]
        shade = grid[0, CHANNELS["shade"]]

        B, C, H, W = grid.shape

        for plant, rule in PLANT_RULES.items():
            idx = get_plant_channel_index(plant)
            
            score = compute_suitability(grid[0], plant, batch_dim=False)

            # Flatten
            flat_score = score.flatten()

            # Normalize to get probabilities
            prob = flat_score / (flat_score.sum() + 1e-8)

            selected = []

            for _ in range(num_seeds_per_plant):
                attempts = 0
                while attempts < 1000:
                    idx_flat = torch.multinomial(prob, 1).item()
                    row = idx_flat // W
                    col = idx_flat % W

                    # Check if too close
                    too_close = False
                    for r0, c0 in selected:
                        if (abs(row - r0) < min_dist) and (abs(col - c0) < min_dist):
                            too_close = True
                            break

                    if not too_close:
                        grid[0, idx, row, col] = 1.0
                        selected.append((row, col))
                        break

                    attempts += 1    

