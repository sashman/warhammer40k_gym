import math
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Warhammer40kEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    phase_mapping = {
        0: "Movement",
        1: "Shooting",
        2: "Charging",
        3: "Fighting",
    }

    def __init__(self, render_mode=None, size=50):
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 48)
        
        self.agent_max_movement = 6
        self.opponent_max_movement = 6

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
                # "turn": spaces.Discrete(2) 

            }
        )

        # angle of movement, fraction of max movement
        self.action_space = spaces.Box(
                np.array([0, 0]).astype(np.float32),
                np.array([360, +1]).astype(np.float32),
            )

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
            "phase": self._current_phase
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

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

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
        if self._current_phase >= len(self.phase_mapping.keys()) - 1:
            self._current_phase = 0    
        else:
            self._current_phase += 1

    def step(self, action):
        
        if(self._current_phase == 0):
            angle = math.radians(action[0])
            magnitude = action[1] * self.agent_max_movement
            direction = self._get_direction_from_polar(angle, magnitude)
            
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        distance = np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )

        if distance < self.target_radius:
            reward = 1
        else:
            reward = 1 / distance

        self._opponent_actions()

        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self._proceed_to_next_phase()
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self._target_location + 0.5) * pix_square_size,
            pix_square_size * self.target_radius,
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Now we draw the opponent
        pygame.draw.circle(
            canvas,
            (127, 0, 127),
            (self._opponent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        img = self.font.render(self.phase_mapping[self._current_phase], True, (100, 100, 100))
        canvas.blit(img, (10, 10))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
