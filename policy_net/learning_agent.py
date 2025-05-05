import torch
import random
from nca.constsants import PLANT_RULES, H, W, CHANNELS
from nca.nca_model import get_species_features_tensor
from nca.suitability import compute_suitability
class LearningAgent:
    def __init__(self, agent_id, policy_net, available_species=None, start_quadrant="top_left", steps_per_turn=10, agent_mask=None):
        self.agent_id = agent_id
        self.policy_net = policy_net
        self.available_species = available_species
        self.start_quadrant = start_quadrant  # <--- NEW
        self.quadrant_mask =  torch.zeros((1, H, W), device='cuda')
        self.saved_log_probs = []
        self.rewards = []
        self.save_interval = steps_per_turn
        self.agent_mask = agent_mask
        self.step_counter = 0
        self.player_number = 0
        self.species_list = []
        self.species_used = []
        self.quadrant_penalty = 0
        self.species_penalty = 0
        self.suitability_reward = 0
        self.diversity_reward = 0
        self.growth_reward = 0
        
        self.round_quadrant_penalty = 0

        
        self.quadrant_skill = 0
        self.species_skill = 0
        self.suitability_skill = 0
        self.growth_skill = 0
        self.training_stage = 0
        self.diversity_skill = 0
        # If no species list is provided, sample from PLANT_RULES
        if available_species is not None:
            self.available_species = available_species
        else:
            self.available_species = random.sample(list(PLANT_RULES.keys()), k=4)
    def check_curriculum(self):
        if (self.training_stage == 0):
            if (self.quadrant_penalty) == 0:
                self.quadrant_skill += 1
                if (self.quadrant_skill > 3):
                    print(f"Agent {self.agent_id} has learned quadrants!!")
                    self.training_stage = 1

        if (self.training_stage == 1):
           if (self.species_penalty == 0):
               self.species_skill += 1
               if (self.species_skill > 5):
                    print(f"Agent {self.agent_id} has learned species!!")
                    self.training_stage = 2
                
        if (self.training_stage == 2):
           if (self.suitability_reward  > 1200):
               self.suitability_skill += 1
               if (self.suitability_skill > 10):
                    print(f"Agent {self.agent_id} has learned suitability!!")
                    self.training_stage = 3
        if (self.training_stage == 3):
           if (self.diversity_reward  >= 1500):
               self.diversity_skill += 1
               if (self.diversity_skill > 10):
                    print(f"Agent {self.agent_id} has learned diversity!!")
                    self.training_stage = 4
        if (self.training_stage == 4):
            if (self.growth_reward > 2000):
                self.growth_skill += 1
                if (self.growth_skill > 10):
                    print(f"Agent {self.agent_id} has leanred growth!!")
                    self.training_stage = 5



    def log_and_reset_loss(self):
        print(f"Agent {self.agent_id} Total: {int(self.quadrant_penalty + self.species_penalty + self.suitability_reward +  self.diversity_reward)} Quad Pen: {int(self.quadrant_penalty)} Species Pen: {int(self.species_penalty)}  Suit Rew: {int(self.suitability_reward)} Diversity  Rew {int(self.diversity_reward)}")
        self.quadrant_penalty = 0
        self.species_penalty = 0
        self.suitability_reward = 0
        self.diversity_reward = 0
        self.growth_reward = 0
        self.species_used = []
    def randomize_species(self,species_sample_size):
        self.available_species = random.sample(list(PLANT_RULES.keys()), k=species_sample_size)

    def choose_action(self, grid, species_features):
        grid = grid.detach()


        ownership_flags = torch.tensor(
            [1.0 if ((self.player_number == 1 and i >= 5) or (self.player_number == 2 and i < 5)) else 0.0
            for i in range(10)],
            device=species_features.device
        ).unsqueeze(1)  # shape: [num_species, 1]

        species_features = torch.cat([species_features, ownership_flags], dim=1)
        species_logits, location_logits = self.policy_net(grid, species_features)

        round_rewards = 0

        if not torch.isfinite(location_logits).all() or not torch.isfinite(species_logits).all():
            print(f"[Agent {self.agent_id}] Invalid logits detected.")
            round_rewards -= 1
            return None


        species_dist = torch.distributions.Categorical(logits=species_logits)
        location_dist = torch.distributions.Categorical(logits=location_logits)
        species_idx = None
        species_max = float("-inf")

        # for i, species in enumerate(self.available_species):
        #     logit = species_logits[0, i].item()
        #     if logit > species_max:
        #         species_max = logit
        #         species_idx = i  # this index maps back to self.available_species[species_idx]

        species_idx = species_dist.sample()
        species_id = species_idx.item()

        if species_idx is None:
            # Fallback: use first available
            print(f"[Agent {self.agent_id}] No valid species logits — falling back to default")
            species_name = self.available_species[0]
            species_idx = next(idx for name, idx in CHANNELS["plants"] if name == species_name)

        round_species_penalty = 0
        round_diversity_reward = 0
        if ((self.player_number == 1 and species_id >= 5) or (self.player_number == 2 and species_id < 5)):
            self.species_penalty -= 1000
            round_species_penalty = -1000
        else:
            if species_id not in self.species_used: 
                self.species_used.append(species_id)
                round_diversity_reward += 500
                self.diversity_reward += 500


        species_name = self.species_list[species_id]

    
        location_idx = location_dist.sample()


        B, HW = location_logits.shape
        pooled_H = pooled_W = int(HW ** 0.5)
        flat_idx = location_idx.item()
        pooled_row = flat_idx // pooled_W
        pooled_col = flat_idx % pooled_W
        scale = grid.shape[2] // pooled_H
        row = pooled_row * scale + scale // 2
        col = pooled_col * scale + scale // 2

        suitability = compute_suitability(grid, species_name)
        suitability_score = suitability[0,row,col]
        self.suitability_reward  += suitability_score * 200
        round_suitability_reward = suitability_score * 200

        log_prob = species_dist.log_prob(species_idx) + location_dist.log_prob(location_idx)
        self.saved_log_probs.append(log_prob)
        
        round_rewards = 0
        if (self.training_stage == 1):
            round_rewards = round_species_penalty
        if (self.training_stage == 2):
            round_rewards = round_species_penalty + round_suitability_reward
        if (self.training_stage == 3):
            round_rewards = round_species_penalty * .2 + round_suitability_reward * .2 + round_diversity_reward
        if (self.training_stage == 4):
            round_rewards = round_species_penalty + round_suitability_reward + round_diversity_reward

        self.rewards.append(round_rewards)
        
        #species_name = self.available_species[species_idx]
        action = (self.agent_id, species_id, row, col)

        return action

    
    def is_legal_move(self, row, col, ownership_grid, quadrant_mask):
        in_quadrant = quadrant_mask[row, col] > 0.5
        owns_cell = (ownership_grid[0, row, col] == self.agent_id)
        return in_quadrant or owns_cell
    
