import torch
import random
from nca.constsants import PLANT_RULES, H, W, CHANNELS
from nca.nca_model import get_species_features_tensor
from nca.suitability import compute_suitability
from collections import deque
class LearningAgent:
    def __init__(self, agent_id, policy_net, available_species=None, start_quadrant="top_left", steps_per_turn=10):
        self.agent_id = agent_id
        self.policy_net = policy_net
        self.available_species = available_species
        self.base_species = available_species
        self.start_quadrant = start_quadrant
        self.quadrant_mask =  torch.zeros((1, H, W), device='cuda')
        self.saved_log_probs = []
        self.rewards = []
        self.save_interval = steps_per_turn
        self.step_counter = 0
        self.player_number = 0
        self.species_list = []
        self.row_used = []
        self.column_used = []
        self.quadrant_penalty = 0
        self.species_penalty = 0
        self.redunancy_penalty = 0

        self.suitability_reward = 0
        self.diversity_reward = 0
        self.growth_reward = 0
        self.growth_buffer = deque(maxlen=20)
        
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

        if self.training_stage == 0:
            if len(self.growth_buffer) == 20:
                successes = sum(1 for g in self.growth_buffer if g > 50)
                success_rate = successes / 20.0
                if success_rate >= 0.9:
                    print(f"Agent {self.agent_id} has learned growth!!")
                    #self.training_stage = 1


        if (self.training_stage == 1):
            if (self.quadrant_penalty) == 0:
                self.quadrant_skill += 1
                if (self.quadrant_skill > 3):
                    print(f"Agent {self.agent_id} has learned quadrants!!")
                    self.training_stage = 2

        if (self.training_stage == 2):
           if (self.species_penalty == 0):
               self.species_skill += 1
               if (self.species_skill > 5):
                    print(f"Agent {self.agent_id} has learned species!!")
                    self.training_stage = 3
                
        if (self.training_stage == 3):
           if (self.suitability_reward  >= 2000):
               self.suitability_skill += 1
               if (self.suitability_skill > 50):
                    print(f"Agent {self.agent_id} has learned suitability!!")
                    self.training_stage = 4
        if (self.training_stage == 4):
           if (self.suitability_reward  >= 2000):
               self.diversity_skill += 1
               if (self.diversity_skill > 30):
                    print(f"Agent {self.agent_id} has learned diversity!!")
                    self.training_stage = 5
        if (self.training_stage == 5):
            if (self.growth_reward > 2000):
                self.growth_skill += 1
                if (self.growth_skill > 10):
                    print(f"Agent {self.agent_id} has leanred growth!!")
                    self.training_stage = 6



    def log_and_reset_loss(self):
        self.growth_buffer.append(self.growth_reward)
        #print(f"Agent {self.agent_id} Stage {self.training_stage} Growth: {int(self.growth_reward)} Quad Pen: {int(self.quadrant_penalty)} Species Pen: {int(self.species_penalty)}  Suit Rew: {int(self.suitability_reward)} Diversity  Rew {int(self.diversity_reward)}")
        self.quadrant_penalty = 0
        self.species_penalty = 0
        self.redunancy_penalty = 0
        self.suitability_reward = 0
        self.diversity_reward = 0
        self.growth_reward = 0
        self.species_used = []
        self.row_used = []
        self.column_used = []
    def randomize_species(self,species_sample_size):
        self.available_species = random.sample(list(PLANT_RULES.keys()), k=species_sample_size)


    def mask_opponent_species(self, grid, agent_species_ids):
        masked_grid = grid.clone()
        for species_id, (_, channel_idx) in enumerate(CHANNELS["plants"]):
            if species_id not in agent_species_ids:
                masked_grid[0, channel_idx] = 0.0
        return masked_grid

    def choose_action(self, grid, species_features):
        grid = grid.detach()

        if self.player_number == 1:
            agent_species_ids = [6,7,8,9,10]
        elif self.player_number == 2:
            agent_species_ids = [11, 12,13,14,15]

        if (self.training_stage == 0):
            masked_grid = grid #self.mask_opponent_species(grid, agent_species_ids)
        else:
            masked_grid = grid

        ownership_flags = torch.tensor(
            [1.0 if ((self.player_number == 1 and i < 5) or (self.player_number == 2 and i >=  5)) else 0.0
            for i in range(10)],
            device=species_features.device
        ).unsqueeze(1)  # shape: [num_species, 1]
        species_features = torch.cat([species_features, ownership_flags], dim=1)
        print("============================")
        print(species_features.shape)
        print(masked_grid.shape)
        species_logits, location_logits = self.policy_net(masked_grid, species_features)

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

        diversity_score = 0
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
                # self.species_used.append(species_id)
                diversity_score += 1

        species_name = self.species_list[species_id]

    
        location_idx = location_dist.sample()


        B, HW = location_logits.shape
        # output_H = grid.shape[2] // 2  # Since we removed the second pooling layer
        # output_W = grid.shape[3] // 2
        # flat_idx = location_idx.item()
        # row = (flat_idx // output_W) * 2 + 1  # +1 centers it in the 2×2 patch
        # col = (flat_idx % output_W) * 2 + 1
        output_H = grid.shape[2]
        output_W = grid.shape[3]
        flat_idx = location_idx.item()
        row = flat_idx // output_W
        col = flat_idx % output_W
        if row not in self.row_used: 
            self.row_used.append(row)
            diversity_score += 1
            
        if col not in self.column_used: 
            self.column_used.append(row)
            diversity_score += 1

        # Get the species channel index directly
        species_channel = CHANNELS["plants"][species_id][1]
        # Extract local 3×3 patch
        local_patch = grid[0, species_channel,
                        max(row - 1, 0):min(row + 2, H),
                        max(col - 1, 0):min(col + 2, W)]

        # Compute average density
        local_density = local_patch.mean()

        # Penalize if species is already present in that area
        round_redundancy_penalty = 0
        if local_density > 0.1:  # threshold: tweak as needed
            round_redundancy_penalty = -1000
            self.redunancy_penalty += round_redundancy_penalty


        suitability = compute_suitability(grid, species_name)
        suitability_score = suitability[0,row,col]
        round_diversity_reward = 0
        round_suitability_reward = 0

        len_species_bonus = (len(self.species_used) -1) * 200
        round_diversity_reward += len_species_bonus 
        self.diversity_reward += len_species_bonus 
        self.suitability_reward  += suitability_score * 400 
        round_suitability_reward = suitability_score * 400  

        log_prob = species_dist.log_prob(species_idx) + location_dist.log_prob(location_idx)
        self.saved_log_probs.append(log_prob)
        
        round_rewards = 0
        if (self.training_stage < 2):
            round_rewards = round_species_penalty
        if (self.training_stage == 2):
            round_rewards = round_species_penalty + round_suitability_reward + round_redundancy_penalty
        if (self.training_stage == 3):
            round_rewards = round_species_penalty + round_suitability_reward + round_diversity_reward + round_redundancy_penalty
        if (self.training_stage >= 4):
            round_rewards = round_species_penalty*.2 + round_suitability_reward*.2 + round_diversity_reward + round_redundancy_penalty

        self.rewards.append(round_rewards)
        
        action = (self.agent_id, species_id, row, col)
        print(action)
        return action

    
    def is_legal_move(self, row, col, ownership_grid, quadrant_mask):
        in_quadrant = quadrant_mask[row, col] > 0.5
        owns_cell = (ownership_grid[0, row, col] == self.agent_id)
        return in_quadrant or owns_cell
    
    def apply_game_end_reward(self, total_score):
        num_actions = len(self.rewards)
        if num_actions == 0:
            return  # No actions to assign rewards to

        per_action_reward = total_score / num_actions
        for i in range(num_actions):
            self.rewards[i] += per_action_reward
        self.growth_reward += total_score  # For logging/debuggin
    
