import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import numpy as np

def animate_species_ownership_with_static_layers(environment, elevation_static, soil_static, steps=None):
    from nca.constsants import CHANNELS, H, W

    if steps is None:
        steps = len(environment.frames)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.tight_layout()

    ims = []

    # Main animated species × ownership
    im_species_owner = axs[0].imshow(torch.zeros((H, W, 3)), animated=True)
    axs[0].axis('off')
    axs[0].set_title("Species + Ownership")

    # Static soil type map
    soil_layers = torch.stack(list(soil_static.values()))  # [4, 1, H, W]
    soil_layers = soil_layers.squeeze(1)  # [4, H, W]

    soil_np = soil_layers.argmax(0).cpu().numpy()  # [H, W]

    axs[1].imshow(soil_np, cmap='Set3')
    axs[1].axis('off')
    axs[1].set_title("Soil Type")

    # Static elevation map
    elevation_np = elevation_static.squeeze(0).cpu().numpy()  # [H, W]
    axs[2].imshow(elevation_np, cmap='terrain', vmin=0, vmax=1)
    axs[2].axis('off')
    axs[2].set_title("Elevation")

    ims.append(im_species_owner)

    # Prepare coloring
# Prepare coloring
    plant_names = [name for name, _ in CHANNELS["plants"]]
    plant_channels = [idx for _, idx in CHANNELS["plants"]]

    colors = plt.cm.tab10(np.linspace(0, 1, len(plant_channels)))[:, :3]  # RGB only
    owner_tints = np.array([
        [1.0, 1.0, 1.0],    # Unclaimed (-1)
        [1.2, 1.0, 1.0],    # Agent 0
        [1.0, 1.2, 1.0],    # Agent 1
        [1.0, 1.0, 1.2],    # Agent 2
        [1.2, 1.2, 1.0],    # Agent 3
        # Extend if more agents
    ])

    def update(i):
        ownership = environment.frames[i].numpy()    # [H, W]
        plants = environment.grid_frames[i].numpy()  # [C, H, W]

        # Pick dominant species at each location
        plant_values = plants[plant_channels]  # [num_species, H, W]
        dominant_species = plant_values.argmax(0)  # [H, W]
        dominant_strength = plant_values.max(0)    # [H, W], how strong the best species is

        # Build base color purely from ownership
        base_color = np.zeros((H, W, 3), dtype=np.float32)

        # Red for agent 0 (ownership == 0)
        base_color[ownership == 0, 0] = 1.0  # Red channel

        # Blue for agent 1 (ownership == 1)
        base_color[ownership == 1, 2] = 1.0  # Blue channel

        # Optionally, you can add more owner colors if needed.

        # Multiply the color by species presence
        intensity = dominant_strength / (dominant_strength.max() + 1e-8)  # normalize
        intensity = np.clip(intensity, 0, 1)

        combined = base_color * intensity[..., None]  # apply intensity per pixel

        im_species_owner.set_data(combined)
        axs[0].set_title(f"Species + Ownership - Step {i}")

        return ims


    ani = animation.FuncAnimation(fig, update, frames=steps, interval=300, blit=False)
    plt.close()
    return ani
