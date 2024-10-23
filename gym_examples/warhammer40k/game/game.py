
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
        
        model_a = Model("model_a", unit=unit_a, agent=Agent())
        model_b = Model("model_b", unit=unit_b, agent=Agent())
        
        unit_a.models = [model_a]
        unit_b.models = [model_b]
        
        player_a.units = [unit_a]
        player_b.units = [unit_b]
        
        world = World(players = [player_a, player_b])
        
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
                
        # add landmarks
        # world.landmarks = [Landmark() for i in range(1)]
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.name = 'landmark %d' % i
        #     landmark.collide = False
        #     landmark.movable = False
        
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