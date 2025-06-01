import numpy as np
import torch
import scipy.ndimage as ndi

from nca.constsants import CHANNELS, NUM_CHANNELS, PLANT_RULES, PLANT_GROUPS, SOIL_TYPES
from nca.suitability import compute_suitability
from nca.constsants import W, H
def generate_elevation_map(h, w, passes=4, falloff=True, variety_type="default"):
    elevation = np.zeros((h, w), dtype=np.float32)

    for i in range(passes):
        freq_scale = 2 ** i
        noise = np.random.rand(h, w)
        sigma = 3 * freq_scale
        blurred = ndi.gaussian_filter(noise, sigma=sigma)
        elevation += blurred / freq_scale

    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())

    y, x = np.ogrid[:h, :w]
    cx, cy = h / 2, w / 2

    if falloff:
        if variety_type == "default":
            d = np.sqrt((x - cx)**2 + (y - cy)**2)
            radial = 1 - np.clip(d / (0.9 * max(h, w) / 2), 0, 1)
            ridge_noise = np.random.rand(h, w)
            ridge_lines = ndi.gaussian_filter(ridge_noise, sigma=10)
            ridge_lines = (ridge_lines - ridge_lines.min()) / (ridge_lines.max() - ridge_lines.min())
            ridge_enhancement = 1 - np.abs(ridge_lines - 0.5) * 2
            falloff_mask = radial * ridge_enhancement
            elevation *= falloff_mask

        elif variety_type == "valleys":
            ridge_noise = np.random.rand(h, w)
            ridge_blur = ndi.gaussian_filter(ridge_noise, sigma=10)
            ridge_mask = 1 - (ridge_blur - ridge_blur.min()) / (ridge_blur.max() - ridge_blur.min())
            elevation *= ridge_mask

        elif variety_type == "ridge":
            ridge = 1 - np.clip(np.abs(x - cx) / (0.5 * w), 0, 1)
            elevation *= ridge

        elif variety_type == "rolling":
            y_noise = np.sin(np.linspace(0, 4 * np.pi, h))[:, None]
            x_noise = np.cos(np.linspace(0, 4 * np.pi, w))[None, :]
            rolling = y_noise + x_noise + np.random.rand(h, w) * 0.2
            rolling = (rolling - rolling.min()) / (rolling.max() - rolling.min())
            elevation = (elevation + rolling) / 2.0

        elif variety_type == "badlands":
            erosion_noise = np.random.rand(h, w)
            erosion_detail = ndi.gaussian_filter(erosion_noise, sigma=1)
            edges = np.abs(ndi.sobel(erosion_detail))
            elevation *= (1 - edges * 0.8)

        elif variety_type == "islands":
            for _ in range(10):
                cx, cy = np.random.randint(0, w), np.random.randint(0, h)
                radius = np.random.randint(10, 20)
                Y, X = np.ogrid[:h, :w]
                dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                bump = np.clip(1 - dist / radius, 0, 1)
                elevation += bump * np.random.uniform(0.3, 0.6)
            elevation = ndi.gaussian_filter(elevation, sigma=3)

    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
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

def generate_training_world(H, W, seed_plants = True, total_plant_species = 10, seed_smart = True, species_list=None):

    variety_options = ["default", "offcenter", "ridge", "basin"]
    variety_idx = np.random.randint(len(variety_options))
    variety_type = variety_options[variety_idx]
    elevation = elevation = generate_elevation_map(H, W, passes=4, falloff=True, variety_type = variety_type)
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
            seed_plants_smart(grid, species_list)  # Add this line
        else:
            seed_plants_random(grid, 8)
    return grid.to("cuda"), variety_idx
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
def seed_plants_smart(grid, species_list, num_seeds_per_plant=1, min_dist=5):
        elevation = grid[0, CHANNELS["elevation"]]
        shade = grid[0, CHANNELS["shade"]]

        B, C, H, W = grid.shape

        for plant in species_list:
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

