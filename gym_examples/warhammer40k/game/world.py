from gym_examples.warhammer40k.constants import GamePhase


class World(object):
    def __init__(self, players):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        
        self.width = 6
        self.heigh = 4
        
        self.current_phase = GamePhase.COMMAND
        self.players = players
        self.units = self.flatten([player.units for player in self.players])
        self.models = self.flatten([unit.models for unit in self.units])
        self.agents = [model.agent for model in self.models]
        
        self.active_player = None
        
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

    # update state of the world
    def step(self):
        
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
            
            
    def flatten(self, xss):
        return [x for xs in xss for x in xs]
        