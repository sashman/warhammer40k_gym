PHASE_MAPPING = {
    0: "movement",
    1: "shooting",
    2: "charging",
    3: "fighting",
}

from enum import Enum

class GamePhase(Enum):
    COMMAND = 0
    MOVEMENT = 1
    SHOOTING = 2
    CHARGING = 3
    FIGHTING =4