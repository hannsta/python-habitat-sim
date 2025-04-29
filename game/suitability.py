
from game.constsants import CHANNELS, PLANT_RULES
import torch


def compute_suitability(grid, plant_name, batch_dim=True):
    """
    Compute habitat suitability score for a given plant.

    Args:
        grid: tensor of shape [B, C, H, W] if batch_dim=True, else [C, H, W]
        plant_name: plant species string, e.g., 'grass_0'
        batch_dim: whether the grid includes batch dimension or not

    Returns:
        suitability: tensor of shape [B, H, W] or [H, W]
    """
    rule = PLANT_RULES[plant_name]
    soil_idx = CHANNELS["soil"][rule["soil"]]

    if batch_dim:
        soil_mask = grid[:, soil_idx]
        elevation = grid[:, CHANNELS["elevation"]]
        shade = grid[:, CHANNELS["shade"]]
    else:
        soil_mask = grid[soil_idx]
        elevation = grid[CHANNELS["elevation"]]
        shade = grid[CHANNELS["shade"]]

    # Elevation preference
    if rule["elevation"] == "low":
        elev_pref = 1 - elevation
    elif rule["elevation"] == "medium":
        elev_pref = 1 - (elevation - 0.5).abs()
    elif rule["elevation"] == "high":
        elev_pref = elevation
    else:
        elev_pref = torch.ones_like(elevation)

    # Shade preference
    if rule["shade"] == "low":
        shade_pref = 1 - shade
    elif rule["shade"] == "high":
        shade_pref = shade
    else:
        shade_pref = torch.ones_like(shade)

    suitability = soil_mask * elev_pref * shade_pref
    suitability = torch.clamp(suitability, 0.0, 1.0)

    return suitability


