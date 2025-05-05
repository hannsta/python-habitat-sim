
from nca.constsants import CHANNELS, PLANT_RULES, SOIL_TYPES
import torch


def compute_suitability(grid, plant_name, batch_dim=True):
    """
    Compute habitat suitability score for a given plant using binary compatibility flags.
    """
    rule = PLANT_RULES[plant_name]

    # --- Soil Compatibility ---
    soil_vals = []
    for i, include in enumerate(rule["soil_types"]):
        if include:
            soil_key = SOIL_TYPES[i]
            soil_idx = CHANNELS["soil"][soil_key]
            soil_vals.append(grid[:, soil_idx] if batch_dim else grid[soil_idx])
    if soil_vals:
        soil_mask = torch.stack(soil_vals).sum(0).clamp(0, 1)  # handles overlaps
    else:
        soil_mask = torch.zeros_like(grid[:, 0] if batch_dim else grid[0])

    # --- Elevation Compatibility ---
    elevation = grid[:, CHANNELS["elevation"]] if batch_dim else grid[CHANNELS["elevation"]]
    elev_mask = torch.zeros_like(elevation)
    for i, allow in enumerate(rule["elevation"]):
        if allow:
            if i == 0:  # low
                elev_mask += (1 - elevation)
            elif i == 1:  # med
                elev_mask += 1 - (elevation - 0.5).abs()
            elif i == 2:  # high
                elev_mask += elevation
    elev_mask = torch.clamp(elev_mask, 0.0, 1.0)

    # --- Shade Compatibility ---
    shade = grid[:, CHANNELS["shade"]] if batch_dim else grid[CHANNELS["shade"]]
    shade_mask = torch.zeros_like(shade)
    for i, allow in enumerate(rule["shade"]):
        if allow:
            if i == 0:  # low
                shade_mask += (1 - shade)
            elif i == 1:  # med
                shade_mask += 1 - (shade - 0.5).abs()
            elif i == 2:  # high
                shade_mask += shade
    shade_mask = torch.clamp(shade_mask, 0.0, 1.0)

    # Combine
    suitability = soil_mask * elev_mask * shade_mask
    return torch.clamp(suitability, 0.0, 1.0)


