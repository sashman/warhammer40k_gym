
from gym_examples.warhammer40k.game.agent import Agent
from gym_examples.warhammer40k.game.model import Model
from gym_examples.warhammer40k.game.player import Player
from gym_examples.warhammer40k.game.unit import Unit
from gym_examples.warhammer40k.game.world import World
import numpy as np


class Game():
    
    
    def make_world(self):
        
        player_a = Player("player_a")
        player_b = Player("player_b")
        
        unit_a = Unit("unit_a", player=player_a)
        unit_b = Unit("unit_b", player=player_b)
        
        model_a = Model("model_a", unit=unit_a)
        model_b = Model("model_b", unit=unit_b)
        
        agents = [
            Agent("agent_a", model=model_a, unit=unit_a, player=player_a),
            Agent("agent_b", model=model_b, unit=unit_b, player=player_b),
            # Agent("agent_c", model=model_c, unit=unit_a, player=player_a),
            # Agent("agent_d", model=model_d, unit=unit_b, player=player_b)
        ]
        
        unit_a.models = [model_a]
        unit_b.models = [model_b]
        
        player_a.units = [unit_a]
        player_b.units = [unit_b]
        
        
        world = World(players = [player_a, player_b], agents = agents)
        
        # for i, agent in enumerate(world.agents):
        #     agent.name = 'agent %d' % i
                        
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
                    
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)        
            

    def reward(self, agent, world):
        return 0

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return entity_pos
    