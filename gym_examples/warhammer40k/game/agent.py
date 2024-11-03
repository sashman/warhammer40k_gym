import math
import numpy as np
from gym_examples.warhammer40k.game.coord_util import get_direction_from_polar
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
        
        self.has_shot_this_phase = False
        
        self.has_been_init = False
        
    def get_location(self):
        return self.state.p_pos
        
    def set_location(self, location):
        self.state.p_pos = location
        
    def move(self, azimuth, distance, max_height, max_width):
        
        angle = math.radians(azimuth)
        magnitude = distance * self.model.max_movement
        direction = get_direction_from_polar(angle, magnitude)
                
        # We use `np.clip` to make sure we don't leave the grid
        self.state.p_pos = np.clip(
                self.get_location() + direction, 0, max_height - 1
            )
    
        self.has_moved_this_phase = True
        
    def shoot(self):
        self.has_shot_this_phase = True
        
    def __str__(self):
        return f"Agent {self.name} Player {self.player.id}"