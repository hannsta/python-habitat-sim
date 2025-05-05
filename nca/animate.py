from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import time
from nca.constsants import CHANNELS, H, W
from nca.generate_map import generate_training_world
from nca.nca_model import update_shade, get_species_features_tensor

from matplotlib import animation
import matplotlib.pyplot as plt
import torch
import time
from nca.constsants import CHANNELS, H, W
from nca.generate_map import generate_training_world
from nca.nca_model import update_shade, get_species_features_tensor

def animate_full_ecosystem(model, initial_grid, elevation_static, soil_static, shade_static, steps=60):
    fig, axs = plt.subplots(4, 4, figsize=(16, 10))
    plt.tight_layout()

    # Pre-initialize images
    ims = []
    for ax in axs.flatten():
        im = ax.imshow(torch.zeros((H, W)), animated=True, cmap="viridis", vmin=0, vmax=1)
        ax.axis('off')
        ims.append(im)

    plant_channels = CHANNELS["plants"]
    plant_names = [name for name, _ in plant_channels]

    def update(i):
        update_shade(initial_grid)
        species_features = get_species_features_tensor(device=initial_grid.device)
        initial_grid.data = model(initial_grid, species_features)

        # Restore static elevation, soil, and shade
        initial_grid[:, CHANNELS["shade"]] = shade_static
        initial_grid[:, CHANNELS["elevation"]] = elevation_static
        for idx, value in soil_static.items():
            initial_grid[:, idx] = value

        grid_cpu = initial_grid[0].detach().cpu()

        for plot_idx, (plant_name, channel_idx) in enumerate(plant_channels):
            data = grid_cpu[channel_idx]
            ims[plot_idx].set_data(data)
            axs[plot_idx // 4, plot_idx % 4].set_title(f"{plant_name} - Step {i}")

        axs[3, 0].imshow(grid_cpu[0:4].argmax(0), cmap="Set3")
        axs[3, 0].set_title("Soil Type")

        axs[3, 1].imshow(grid_cpu[CHANNELS["elevation"]], cmap="terrain", vmin=0, vmax=1)
        axs[3, 1].set_title("Elevation")

        axs[3, 2].imshow(grid_cpu[CHANNELS["shade"]], cmap="bone", vmin=0, vmax=1)
        axs[3, 2].set_title("Shade")

        axs[3, 3].axis("off")

        return ims

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=300, blit=False)
    plt.close()
    return ani
