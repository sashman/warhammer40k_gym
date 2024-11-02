#!/usr/bin/env python
from pprint import pp
import os,sys

from gym_examples.envs.multi_agent_warhammer40k import MultiAgentWarhammer40k
from gym_examples.warhammer40k.game.game import Game
from gym_examples.warhammer40k.game.policy import InteractivePolicy
sys.path.insert(1, os.path.join(sys.path[0], '..'))

def main():
    
    # load scenario from script
    scenario = Game()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentWarhammer40k(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.number_of_agents)]
    
    # execution loop
    obs_n = env.reset()
    while True:
    # for i in range(0, 10):
        

        act_n = []        
        for i, agent in enumerate(env.world.agents):
            # sample a random action from action space
            act_n.append(env.action_space[i][0].sample())
            
        
        pp(act_n)
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        
        # render all agent views
        env.render()
        
        # display rewards
        for agent in env.world.agents:
            print(" GAME STATE " + str(env.world) + " " + agent.name + " reward: %0.3f" % env._get_reward(agent) + " AGENT STATE has_moved_this_phase=" + str(agent.has_moved_this_phase))


if __name__ == '__main__':
    main()