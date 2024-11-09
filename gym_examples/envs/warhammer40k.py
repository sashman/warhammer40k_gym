import math
from gym_examples.warhammer40k.constants import PHASE_MAPPING
from gym_examples.warhammer40k.render import close, render_frame
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Warhammer40kEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, size=50):
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 48)
        
        self.agent_max_movement = 6
        self.opponent_max_movement = 6

        self._agent_shooting_target_locaiton = np.array([])

        self.target_radius = 3

        objective = spaces.Box(0, size - 1, shape=(2,), dtype=int)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "opponent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": objective,
                # Phases
                # 0 Movement
                # 1 Shooting
                # 2 Charging
                # 3 Fighting
                "phase": spaces.Discrete(1),
                # Turn number, which player to act
                # 0 - players turn
                # 1 - opponents turn
                "turn": spaces.Discrete(2) 
            }
        )

        # Indexed by phase
        self.action_space = spaces.Dict({
            # Moving
            0: spaces.Box(low=np.array([0, 0]), high=np.array([360, +1]), dtype=np.float32),
            # Shooting
            1: spaces.Box(low=-1, high=0, dtype=np.int8),
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "agent": self._agent_location, 
            "target": self._target_location, 
            "opponent": self._opponent_location,
            "phase": self._current_phase,
            "turn": self._current_turn,
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _get_direction_from_polar(self, angle_radians, magnitude):
        x = magnitude * np.cos(angle_radians)
        y = magnitude * np.sin(angle_radians)
        direction = np.floor(np.array([x,y])).astype(int)

        return direction

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self._opponent_location = self._agent_location
        while np.array_equal(self._opponent_location, self._target_location):
            self._opponent_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self._current_phase = 0
        self._current_turn = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            render_frame(self)

        return observation, info
    
    def _reset_targets(self):
        self._agent_shooting_target_locaiton = np.array([])

    def _get_agent_shooting_target_location_from_index(self, index):
        return self._opponent_location
    
    def _player_actions(self, action):
        if(self._current_phase == 0):
            angle = math.radians(action[0][0])
            magnitude = action[0][1] * self.agent_max_movement
            direction = self._get_direction_from_polar(angle, magnitude)
                
                # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                    self._agent_location + direction, 0, self.size - 1
                )

        if(self._current_phase == 1):
            shooting_action = action[1]

            if shooting_action >= 0:
                self._agent_shooting_target_locaiton = self._get_agent_shooting_target_location_from_index(0)

    def _opponent_actions(self):

        if(self._current_phase == 0):
            # Move directly towards target at max speed
            diff = self._target_location - self._opponent_location
            distance = np.sqrt(diff[0]**2 + diff[1]**2)
            angle = np.arctan2(diff[1], diff[0])

            possible_distance = np.clip(distance, 0, self.opponent_max_movement)
            direction = self._get_direction_from_polar(angle, possible_distance)
            
            # We use `np.clip` to make sure we don't leave the grid
            self._opponent_location = np.clip(
                self._opponent_location + direction, 0, self.size - 1
            )

    def _proceed_to_next_phase(self):
        if self._current_phase >= len(PHASE_MAPPING.keys()) - 1:
            self._current_phase = 0
            if self._current_turn == 0:
                self._current_turn = 1
            else:
                self._current_turn = 0

            self._reset_targets()
        else:
            self._current_phase += 1

    def step(self, action):
        
        if self.is_player_turn():
            self._player_actions(action)
        elif self.is_opponent_turn():
            self._opponent_actions()

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        distance = np.linalg.norm(
            self._agent_location - self._target_location, ord=1
        )
        if distance < self.target_radius:
            reward = 1
        else:
            reward = 1 / distance

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            render_frame(self)

        self._proceed_to_next_phase()
        return observation, reward, terminated, False, info

    
    def is_player_turn(self):
        return self._current_turn == 0
    
    def is_opponent_turn(self):
        return self._current_turn == 1

    def render(self):
        if self.render_mode == "rgb_array":
            # return self._render_frame()
            return render_frame(self)

    def close(self):
        close(self)