#!/usr/bin/env python
from pprint import pp
from gym_examples.warhammer40k.algorithm.mcts.mcts import mcts
import numpy as np
import os,sys

from gym_examples.envs.multi_agent_warhammer40k import MultiAgentWarhammer40k
from gym_examples.warhammer40k.game.game import Game
from gym_examples.warhammer40k.game.policy import InteractivePolicy
sys.path.insert(1, os.path.join(sys.path[0], '..'))

def single_game(env: MultiAgentWarhammer40k):
    final_reward_n = []
    final_obs_n = []
    # execution loop
    obs_n = env.reset()
    while not env.game_over():
        act_n = []        
        for i, agent in enumerate(env.world.agents):
            
            agent_action_space = env.action_space[i][0]
            # sub sample continuous action space into discrete action space
            # action_space_samples = get_discrete_action_space_samples(agent_action_space)
            
            # # Run MCTS on the environment
            # num_simulations = 1000
            # best_action = mcts(obs_n, env, num_simulations, agent_index=i, action_space=action_space_samples)
            
            # sample a random action from action space
            # action = best_action
            action = agent_action_space.sample()
            
            # action = np.array([270, 1])
            act_n.append(action)
            
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        print("################# obs_n #################")
        pp(obs_n)
        final_reward_n = reward_n
        print("################# reward_n #################")
        pp(reward_n)
        final_reward_n = reward_n
        
        # render all agent views
        env.render()
        
        # display rewards
        for agent in env.world.agents:
            print(" GAME STATE " + str(env.world) + " " + agent.name + " reward: %0.3f" % env._get_reward(agent) + " AGENT STATE has_moved_this_phase=" + str(agent.has_moved_this_phase))
    return final_obs_n, final_reward_n
            
def get_discrete_action_space_samples(action_space):
    # get discrete action space samples
    action_space_samples = []
    for angle in range(0, 360, 5):
        for j in range(1, 100, 10):
            distance = j / 100
            action_space_samples.append([angle, distance])
    return action_space_samples

def main():
    
    # load scenario from script
    scenario = Game()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentWarhammer40k(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done, info_callback=None, shared_viewer = False)
    
    env.init_renderer()
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.number_of_agents)]
    
    final_obs_n, final_reward_n = single_game(env)
    
    pp("################# game over #################")
    pp(final_obs_n)
    
    pp(final_reward_n)


if __name__ == '__main__':
    main()