import torch
import numpy as np
from nca.nca_model import log_channel_sums

def finish_episode(agent, optimizer, grid, gamma=0.10):
    returns = []
    R = 0.0
    print(f"Agent {agent.agent_id} Reward ++++ {agent.rewards}")
    for r in reversed(agent.rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, device='cuda')

    
    returns = torch.tensor(returns, device='cuda')
    advantage = returns
    optimizer.zero_grad()
    losses = [-log_prob * adv for log_prob, adv in zip(agent.saved_log_probs, advantage)]
    loss = torch.stack(losses).sum()

    print(f"Agent {agent.agent_id} Loss ==== {loss.item()}")
    loss.backward()
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
        pre_grid, post_grid = environment.step(actions)
        compute_growth_rewards_from_grid_diff(pre_grid, post_grid, actions, environment.agents)
    scores = environment.get_scores()
    #log_channel_sums(environment.grid)
    return scores, game_actions


def compute_growth_rewards_from_grid_diff(pre_grid, post_grid, actions, agents, base_channel=6, scale=1):
    for agent_id, species_idx, _, _ in actions:
        agent = next(a for a in agents if a.agent_id == agent_id)
        chan = base_channel + species_idx

        delta = post_grid[0, chan] - pre_grid[0, chan]
        growth = torch.clamp(delta, min=0).sum().item()  # Only reward positive growth
        if species_idx not in agent.species_used:
            agent.species_used.append(species_idx)

            agent.rewards[-1] += (growth * scale)
            agent.growth_reward += growth * scale

    
