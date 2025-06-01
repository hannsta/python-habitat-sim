from nca.constsants import CHANNELS, H , W
from nca.nca_model import update_shade, get_species_features_tensor,log_channel_sums
import torch

class Environment:
    def __init__(self, grid, model, ownership_grid, agents, elevation_static, soil_static, shade_static, species_features=None, species_list=None, steps_per_turn=10):
        self.grid = grid
        self.model = model
        self.ownership_grid = ownership_grid
        self.agents = agents
        self.steps_per_turn = steps_per_turn
        self.elevation_static = elevation_static
        self.soil_static = soil_static
        self.shade_static = shade_static
        self.current_turn = 0
        self.frames = []
        self.grid_frames = []
        self.species_features = species_features
        self.quadrant_masks = torch.zeros((len(self.agents), H, W), device=self.grid.device)
        self.species_list = species_list
        for agent in self.agents:
            agent.species_list = species_list     
        self.refresh_quadrant_masks()

    def refresh_quadrant_masks(self):
        for agent in self.agents:
            if agent.start_quadrant == "top_left":
                agent.quadrant_mask[0, :H//2, :W//2] = 1.0
            elif agent.start_quadrant == "bottom_right":
                agent.quadrant_mask[0, H//2:, W//2:] = 1.0

    def get_frames(self):
        return self.frames
    
    def step(self, actions):
        pre_step_grid = self.grid.clone()

        # Apply all actions in-place
        for agent_id, plant_idx, row, col in actions:
            self.grid[0, 6 + plant_idx, row, col] = 1
            self.ownership_grid[0, row, col] = agent_id

        # Cache static layers to local variables
        elevation = self.elevation_static
        shade = self.shade_static
        soil = self.soil_static
        steps = self.steps_per_turn

        with torch.no_grad():
            for _ in range(steps):
                # Grow vegetation
                self.grid = self.model(self.grid, self.species_features)

                # Reset static layers
                self.grid[:, CHANNELS["elevation"]] = elevation
                self.grid[:, CHANNELS["shade"]] = shade
                for idx, val in soil.items():
                    self.grid[:, idx] = val

                # Store for animation and scoring
                self.frames.append(self.ownership_grid[0].detach().cpu().clone())
                self.grid_frames.append(self.grid[0].detach().cpu().clone())

                # Update ownership labels based on plant spread
                self.update_ownership()

        self.current_turn += 1
        return pre_step_grid.clone(), self.grid.clone()


    def update_ownership(self):
        """Optional: Transfer ownership as plants expand."""
        for species_name, idx in CHANNELS["plants"]:
            mask = self.grid[0, idx] > 0.1
            owner = -1
            for agent in self.agents:
                if agent.player_number == 1 and idx < 11:
                    owner = agent.agent_id
                elif agent.player_number == 2 and idx >= 11:
                    owner = agent.agent_id
            if owner != -1:
                self.ownership_grid[0][mask] = owner  # Assume one agent per species for now


    def get_scores(self):
        """Returns a dict of agent_id: controlled area."""
        scores = {agent.agent_id: 0 for agent in self.agents}
        for agent_id in scores.keys():
            scores[agent_id] = (self.ownership_grid[0] == agent_id).float().sum().item()
        return scores

