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


import matplotlib.pyplot as plt
import numpy as np

def plot_agent_metrics(agent_logs, ma_window=10):
    episodes = list(next(iter(agent_logs.values()))["episode"])  # Assumes both agents share episode indices

    # --- Plot 1: Growth Reward (Score) with Moving Average ---
    plt.figure(figsize=(10, 3))
    plt.title("Growth Reward (with Moving Average)")
    for agent_id, log in agent_logs.items():
        growth = np.array(log["growth_reward"])
        ma = np.convolve(growth, np.ones(ma_window)/ma_window, mode='valid')
        plt.plot(episodes, log["growth_reward"], alpha=0.3, label=f"Agent {agent_id} (raw)")
        plt.plot(episodes[ma_window-1:], ma, label=f"Agent {agent_id} (MA {ma_window})")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Training Stage ---
    plt.figure(figsize=(10, 3))
    plt.title("Training Stage Progression")
    for agent_id, log in agent_logs.items():
        plt.plot(episodes, log["training_stage"], label=f"Agent {agent_id} Stage")
        plt.plot(episodes, log["species_set"], label=f"Agent {agent_id} Species Set")
        plt.plot(episodes, log["world_variety"], label=f"Agent {agent_id} Variety")


    plt.xlabel("Episode")
    plt.ylabel("Stage")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 3: Rewards ---
    plt.figure(figsize=(10, 3))
    plt.title("Suitability & Diversity Rewards")
    for agent_id, log in agent_logs.items():
        plt.plot(episodes, log["suitability_reward"], label=f"Agent {agent_id} - Suitability")
        plt.plot(episodes, log["diversity_reward"], label=f"Agent {agent_id} - Diversity")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 4: Penalties ---
    plt.figure(figsize=(10, 3))
    plt.title("Quadrant & Species Penalties")
    for agent_id, log in agent_logs.items():
        plt.plot(episodes, log["quadrant_penalty"], label=f"Agent {agent_id} - Quad")
        plt.plot(episodes, log["species_penalty"], label=f"Agent {agent_id} - Species")
    plt.xlabel("Episode")
    plt.ylabel("Penalty")
    plt.legend()
    plt.tight_layout()
    plt.show()



import pandas as pd
import statsmodels.api as sm
def print_variable_significance(agent_logs):
    import pandas as pd
    import statsmodels.api as sm

    # Combine logs
    df0 = pd.DataFrame({
        "agent": 0,
        "score": agent_logs[0]["score"],
        "world_variety": agent_logs[0]["world_variety"],
        "species_set": agent_logs[0]["species_set"],
        "player_number": agent_logs[0]["player_number"]
    })

    df1 = pd.DataFrame({
        "agent": 1,
        "score": agent_logs[1]["score"],
        "world_variety": agent_logs[1]["world_variety"],
        "species_set": agent_logs[1]["species_set"],
        "player_number": agent_logs[1]["player_number"]
    })

    df = pd.concat([df0, df1], ignore_index=True)

    # One-hot encode categorical variables WITHOUT dropping first
    df_encoded = pd.get_dummies(df, columns=["world_variety", "species_set", "agent", "player_number"], drop_first=False)

    df_encoded = df_encoded.apply(pd.to_numeric, errors="coerce").dropna()

    X = df_encoded.drop(columns=["score"])
    X = sm.add_constant(X).astype(float)
    y = df_encoded["score"].astype(float)

    reg_model = sm.OLS(y, X).fit()
    print(reg_model.summary())

    print("\nCorrelation with score:")
    print(df[["score", "world_variety", "species_set", "player_number"]].corr(numeric_only=True)["score"])


def plot_success_rate(agent_logs, threshold=20, window=20):
    episodes = list(next(iter(agent_logs.values()))["episode"])  # Assumes same episodes

    plt.figure(figsize=(10, 3))
    plt.title(f"Success Rate (Score > {threshold}) over {window}-Episode Window")

    for agent_id, log in agent_logs.items():
        scores = np.array(log["score"])
        # Calculate success (1 if score > threshold, else 0)
        successes = (scores > threshold).astype(int)

        # Rolling sum of successes, normalized to percentage
        rolling_success = np.convolve(successes, np.ones(window, dtype=int), mode='valid') / window * 100

        plt.plot(episodes[window - 1:], rolling_success, label=f"Agent {agent_id}")

    plt.xlabel("Episode")
    plt.ylabel(f"% Success (>{threshold})")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

def plot_species_choices_over_time(agent_logs, species_list, window=20):
    species_id_to_name = {i: name for i, name in enumerate(species_list)}

    for agent_id, log in agent_logs.items():
        choices = log["choices"]  # List of list of (species_id, x, y)
        episodes = log["episode"]

        # Prepare rolling frequency matrix: shape [num_windows, num_species]
        num_species = len(species_list)
        freq_matrix = []

        for i in range(len(choices) - window + 1):
            window_choices = choices[i:i+window]
            flat_species = [sid for sublist in window_choices for (sid, _, _) in sublist]
            counts = Counter(flat_species)
            total = sum(counts.values())

            freqs = [counts.get(sid, 0) / total if total > 0 else 0.0 for sid in range(num_species)]
            freq_matrix.append(freqs)

        freq_matrix = np.array(freq_matrix)  # shape [T, num_species]
        moving_episodes = episodes[window - 1:]

        # Plot
        plt.figure(figsize=(12, 4))
        plt.title(f"Agent {agent_id} - Species Choice Frequency (Window={window})")
        for sid in range(num_species):
            plt.plot(moving_episodes, freq_matrix[:, sid], label=species_id_to_name[sid])
        plt.xlabel("Episode")
        plt.ylabel("Species Usage %")
        plt.legend(loc="upper right", ncol=3, fontsize=8)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display

def plot_dual_axis(agent_logs):
    agent_ids = list(agent_logs.keys())
    metrics = [key for key in agent_logs[agent_ids[0]].keys() if key not in ["episode", "growth_reward"]]

    agent_dropdown = widgets.Dropdown(options=agent_ids, description="Agent:")
    metric_dropdown = widgets.Dropdown(options=metrics, description="Metric:")
    ma_slider = widgets.IntSlider(value=10, min=1, max=50, step=1, description='MA Window:')

    def update_plot(agent_id, selected_metric, ma_window):
        log = agent_logs[agent_id]
        episodes = np.array(log["episode"])
        score = np.array(log["growth_reward"])

        if selected_metric == "choices":
            metric = np.array([c[0][0] for c in log["choices"]])
        else:
            metric = np.array(log[selected_metric]).squeeze()
            if metric.ndim > 1:
                metric = metric.mean(axis=1)  # Or apply another reduction strategy

        fig, ax1 = plt.subplots(figsize=(10, 4))

        # Plot growth reward
        ma_score = np.convolve(score, np.ones(ma_window)/ma_window, mode='valid')
        ax1.plot(episodes, score, alpha=0.3, label="Growth (raw)", color="tab:blue")
        ax1.plot(episodes[ma_window-1:], ma_score, label=f"Growth (MA {ma_window})", color="tab:blue", linestyle="--")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Growth Reward", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor='tab:blue')



        # # Plot secondary metric
        # ax2 = ax1.twinx()
        # ax2.plot(episodes, metric, label=selected_metric, alpha=0.3, color="tab:orange")
        # ax2.set_ylabel(selected_metric, color="tab:orange")
        # ax2.tick_params(axis='y', labelcolor='tab:orange')


        plt.title(f"Agent {agent_id} - Growth vs {selected_metric}")
        fig.tight_layout()
        plt.show()

    out = widgets.interactive_output(
        update_plot,
        {"agent_id": agent_dropdown, "selected_metric": metric_dropdown, "ma_window": ma_slider}
    )

    display(widgets.HBox([agent_dropdown, metric_dropdown, ma_slider]), out)