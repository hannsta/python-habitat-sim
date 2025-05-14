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
            elif agent.start_quadrant == "top_right":
                agent.quadrant_mask[0, :H//2, W//2:] = 1.0
            elif agent.start_quadrant == "bottom_left":
                agent.quadrant_mask[0, H//2:, :W//2] = 1.0

    def get_frames(self):
        return self.frames

    def step(self, actions):
        print(actions)
        pre_step_grid = self.grid.clone()
        """Actions is a list of (agent_id, species_name, row, col)"""
        for agent_id, plant_idx, row, col in actions:
            self.grid[0, 6 + plant_idx, row, col] = 1
            self.ownership_grid[0, row, col] = agent_id
        # Run growth
        for i in range(self.steps_per_turn):
            with torch.no_grad():
                # print("============================")
                # log_channel_sums(self.grid)
                # print("----------------------")
                self.grid = self.model(self.grid, self.species_features) 
                #update_shade(self.grid)

  
                self.grid[:, CHANNELS["elevation"]] = self.elevation_static
                self.grid[:, CHANNELS["shade"]] = self.shade_static
                
                for idx, value in self.soil_static.items():
                    self.grid[:, idx] = value
                

                # log_channel_sums(self.grid)
                # print("============================")
                self.frames.append(self.ownership_grid[0].detach().cpu().clone())
                self.grid_frames.append(self.grid[0].detach().cpu().clone()) 
                self.update_ownership()

        # Restore static channels
        self.current_turn += 1
        return pre_step_grid.clone(), self.grid.clone()  # return before and after state

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

