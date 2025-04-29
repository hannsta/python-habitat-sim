from matplotlib import animation
import matplotlib.pyplot as plt
import torch

from game.constsants import CHANNELS, H, W
from game.generate_map import generate_training_world
from game.nca_model import update_shade

def animate_full_ecosystem(model, initial_grid, elevation_static, soil_static, steps=60):
    fig, axs = plt.subplots(3, 4, figsize=(16, 10))
    plt.tight_layout()

    # Pre-initialize images
    ims = []
    for ax in axs.flatten():
        im = ax.imshow(torch.zeros((H, W)), animated=True, cmap="viridis", vmin=0, vmax=1)
        ax.axis('off')
        ims.append(im)

    plant_names = list(CHANNELS["plants"].keys())

    def update(i):
        update_shade(initial_grid)
        initial_grid.data = model(initial_grid)

        # 🔒 Restore static elevation and soil
        initial_grid[:, CHANNELS["elevation"]] = elevation_static
        for idx, value in soil_static.items():
            initial_grid[:, idx] = value

        grid_cpu = initial_grid[0].detach().cpu()

        for idx, name in enumerate(plant_names):
            data = grid_cpu[CHANNELS["plants"][name]]
            ims[idx].set_data(data)
            axs[idx // 4, idx % 4].set_title(f"{name} - Step {i}")

        axs[2, 0].imshow(grid_cpu[0:4].argmax(0), cmap="Set3")
        axs[2, 0].set_title("Soil Type")

        axs[2, 1].imshow(grid_cpu[CHANNELS["elevation"]], cmap="terrain", vmin=0, vmax=1)
        axs[2, 1].set_title("Elevation")

        axs[2, 2].imshow(grid_cpu[CHANNELS["shade"]], cmap="bone", vmin=0, vmax=1)
        axs[2, 2].set_title("Shade")

        axs[2, 3].axis("off")

        return ims

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=300, blit=False)
    plt.close()
    return ani