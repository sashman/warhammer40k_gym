from gym_examples.warhammer40k.game.entity_state import EntityState
import numpy as np

class PrimaryObjective():
    def __init__(self, id, x, y):
        self.id = id
        self.radius = 3
        self.state = EntityState()
        self.state.p_pos = np.array([x, y])
        
    def get_location(self):
        return self.state.p_pos