import torch
import numpy as np
from nca.nca_model import log_channel_sums

def finish_episode(agent, optimizer, grid, gamma=.10):
    returns = []
    R = 0.0
    for r in reversed(agent.rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, device='cuda')

    # Normalize
    if returns.std() > 1e-6:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    else:
        returns = returns - returns.mean()  #
        
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device='cuda')

    losses=[]

    for log_prob, R in zip(agent.saved_log_probs, returns):
        losses.append(-log_prob * R)

    
    total_loss = torch.stack(losses).sum()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    # Clean up
    agent.rewards.clear()
    agent.saved_log_probs.clear()


def play_game(environment, species_features, max_turns=8, save=False):
    game_actions =  []
    for _ in range(max_turns):
        actions = []
        state = environment.grid.clone()
        ownership_grid = environment.ownership_grid  # <-- pull this once
    
        for agent in environment.agents:
            # Build agent-specific grid with quadrant mask attached
            agent_quadrant = agent.quadrant_mask[0].unsqueeze(0)  # shape [1, H, W]
            augmented_grid = torch.cat([state, agent_quadrant.unsqueeze(0)], dim=1)  # add to channels
            if(_ ==0 and save):
                print("savinggg")
                print(augmented_grid.shape)
                np.save("test_grid.npy", augmented_grid.cpu().numpy())

            action = agent.choose_action(augmented_grid, species_features)
            agent_id, species_idx, row, col = action  # <-- UNPACK action now

            # Check legality
            quadrant_mask = agent.quadrant_mask[0]  # just their own mask
            if (agent.training_stage > 0):
                if not agent.is_legal_move(row, col, ownership_grid, quadrant_mask):
                    penalty = 1000.0
                    agent.rewards[-1] -= penalty
                    agent.quadrant_penalty -= penalty
            actions.append(action)
        game_actions.append(actions)
        environment.step(actions)

    scores = environment.get_scores()
    #log_channel_sums(environment.grid)
    return scores, game_actions
