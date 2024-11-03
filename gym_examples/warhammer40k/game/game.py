
from gym_examples.warhammer40k.game.agent import Agent
from gym_examples.warhammer40k.game.model import Model
from gym_examples.warhammer40k.game.player import Player
from gym_examples.warhammer40k.game.primary_objective import PrimaryObjective
from gym_examples.warhammer40k.game.unit import Unit
from gym_examples.warhammer40k.game.world import World
import numpy as np

RED = (255, 0, 0)
BLUE = (0, 0, 255)

class Game():
    
    
    def make_world(self):
        
        player_a = Player("player_a")
        player_b = Player("player_b")
        
        unit_a = Unit("unit_a", player=player_a, unit_colour=BLUE)
        unit_b = Unit("unit_b", player=player_b, unit_colour=RED)
        
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
        
        primary_objective_a = PrimaryObjective("primary_objective_a", 25, 10)
        primary_objective_b = PrimaryObjective("primary_objective_a", 25, 40)
        
        world = World(players = [player_a, player_b], agents = agents, primary_objectives = [primary_objective_a, primary_objective_b])
        
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        print("reset world")
                    
        # set agent's locations off map
        for agent in world.agents:
            agent.set_location(np.array([-1,-1]))
            

    def reward(self, agent, world):
        return 0

    def observation(self, agent, world):
        # build observation state of world
        
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.entities:
            entity_pos.append(entity.state.p_pos)
            
        # get positions of all primary objectives in this agent's reference frame
        primary_objectives_pos = []
        for primary_objective in world.primary_objectives:
            primary_objectives_pos.append(primary_objective.state.p_pos)
            
        # get distance to all entities in this agent's reference frame
        entity_dist = []
        for entity in world.entities:
            entity_dist.append(np.linalg.norm(agent.state.p_pos - entity.state.p_pos))
            
        # get distance to all primary objectives in this agent's reference frame
        primary_objectives_dist = []
        for primary_objective in world.primary_objectives:
            primary_objectives_dist.append(np.linalg.norm(agent.state.p_pos - primary_objective.state.p_pos))
        
        obs_dict = {
            'entity_pos': entity_pos,
            'entity_dist': entity_dist,
            'primary_objectives_pos': primary_objectives_pos,
            'primary_objectives_dist': primary_objectives_dist
        }
        
        return obs_dict
    