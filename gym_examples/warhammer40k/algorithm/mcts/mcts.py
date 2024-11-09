import copy
from pprint import pp
import numpy as np
from gym_examples.warhammer40k.algorithm.mcts.mcts_node import MCTSNode


def mcts(initial_state, model, num_simulations, agent_index, action_space, exploration_weight=1.0):
    root = MCTSNode(state=initial_state)

    for _ in range(num_simulations):
        model = copy.deepcopy(model)
        model.disable_render()
        
        node = root
        state = initial_state

        # Selection: Traverse the tree to the most promising leaf node
        pp("Selection")
        while node.children:
            node = node.best_child(exploration_weight=exploration_weight)

        # Expansion
        pp("Expansion")
        if not node.is_fully_expanded(action_space):
            
            # create a list of actions for each agent in model
            action_n = [get_action_from_action_space(action_space) for _ in range(len(model.agents))]
            
            next_state, reward, done, _ = model.step(action_n)
            new_node = MCTSNode(state=next_state, parent=node, action=action_n)
            node.children.append(new_node)

            # Simulation (Rollout)
            pp("Simulation")
            sim_reward = rollout(next_state, model, action_space, done)
            
            pp("Backpropagation")
            backpropagate(new_node, sim_reward)

    return root.best_child(0).action  # Choose the best child without exploration factor

def rollout(state, model, action_space, done):
    total_reward = 0
    while not done:
        action_n = [get_action_from_action_space(action_space) for _ in range(len(model.agents))]
        # action = get_action_from_action_space(action_space)  # Random rollout policy
        state, reward, done, _ = model.step(action_n)
        total_reward += reward
    return total_reward

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.value += reward
        node = node.parent

def get_action_from_action_space(action_space):
    
    # random element from action space list
    return action_space[np.random.choice(len(action_space))]