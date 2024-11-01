from gym_examples.warhammer40k.game.entity_state import EntityState
from gym_examples.warhammer40k.game.player import Player

class Agent():
    def __init__(self, name, model, unit, player: Player):
        super(Agent, self).__init__()
        
        # state
        self.state = EntityState()
        
        # action
        self.action = None
        
        # script behavior to execute
        self.action_callback = None
        
        self.name = name
        
        self.model = model
        
        self.unit = unit
        
        self.player = player
        
        self.movable = True
        
        self.has_moved_this_phase = False
        
    def move(self):
        self.has_moved_this_phase = True
        
    def __str__(self):
        return f"Agent {self.name} Player {self.player.id}"