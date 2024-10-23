from gym_examples.warhammer40k.game.entity_state import EntityState

class Agent():
    def __init__(self):
        super(Agent, self).__init__()
        
        # state
        self.state = EntityState()
        
        # action
        self.action = None
        
        # script behavior to execute
        self.action_callback = None
        
        self.name = None
        
        self.movable = True