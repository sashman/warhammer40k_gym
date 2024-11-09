from itertools import groupby
from gym_examples.warhammer40k.game.primary_objective import PrimaryObjective
import numpy as np

from gym_examples.warhammer40k.game.agent import Agent
from gym_examples.warhammer40k.game.world_timing import get_next_world_time_state
from gym_examples.warhammer40k.game.phase import Phase

class World(object):
    def __init__(self, players, agents: list[Agent], primary_objectives: list[PrimaryObjective]):
        
        
        self.landmarks = []
        
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        
        self.width = 50
        self.heigh = 50
        
        self.current_turn = 0
        self.current_player_round = 0
        self.current_phase = Phase.Init
    
        self.players = players
        self.units = self.flatten([player.units for player in self.players])
        self.models = self.flatten([unit.models for unit in self.units])
        self.agents = agents
        self.primary_objectives = primary_objectives
        self.agents_by_player = {}
        
        self.agents_by_player = []
        agents = sorted(agents, key=lambda agent: agent.player.id)
        for k, g in groupby(agents, lambda agent: agent.player.id):
            self.agents_by_player.append(list(g))
            
        
        
    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]
    
    def agents_for_player(self, player_index: int):
        return self.agents_by_player[player_index]

    # update state of the world
    def step(self):
        
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        
        (new_turn, new_player_round, new_phase) = get_next_world_time_state(
            current_phase=self.current_phase,
            current_player_round=self.current_player_round,
            current_turn=self.current_turn,
            players=self.players,
            agents_by_player=self.agents_by_player
        )
        
        if self.current_turn != new_turn:
            self.reset_agents()
            
        self.current_turn = new_turn        
        self.current_player_round = new_player_round
        self.current_phase = new_phase

    def reset_agents(self):
        for agent in self.agents:
            agent.has_moved_this_phase = False
            agent.has_shot_this_phase = False
        # print("\tNext STATE " + str(self))
        
    def game_over(self):
        # game is over after turn 5
        return self.current_turn >= 5
        
    def agent_init_location(self, agent_index):
        return np.array([agent_index % 2 * (self.width - 1), agent_index // 2 * (self.heigh - 1) + self.heigh/2])
            
    def flatten(self, xss):
        return [x for xs in xss for x in xs]
    
    def __str__(self):
        return f"Turn={self.current_turn} Player={self.current_player_round} Phase={self.current_phase}"