import math
import pygame
import numpy as np
from gym_examples.warhammer40k.constants import PHASE_MAPPING

GRID_WIDTH = 1
SHOT_WIDTH = 5

def render_frame(self):
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

        render_objective(canvas, self._target_location, self.target_radius, pix_square_size)

        # Now we draw the agent
        render_model(canvas, (0,0,255), self._agent_location, pix_square_size)

        # Now we draw the opponent
        render_model(canvas, (127,0,127), self._opponent_location, pix_square_size)        

        # Finally, add some gridlines
        render_grid(self, canvas, pix_square_size)

        if not self._agent_shooting_target_locaiton.size == 0 and self._current_phase == 1:
            render_shot(canvas, self._agent_location, self._agent_shooting_target_locaiton, pix_square_size)

        render_phase_label(canvas, self, self._current_phase)
        
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

def render_phase_label(canvas, self, phase):
    turn_label = "opponent:"

    if self.is_player_turn():
        turn_label = "player:" 

    img = self.font.render(f"{turn_label} {PHASE_MAPPING[phase]}", True, (100, 100, 100))
    canvas.blit(img, (10, 10))


def render_shot(canvas, shooter_location, target_location, pix_square_size):
    
    pygame.draw.line(
        canvas,
        (255,127,0),
        (shooter_location +.5) * pix_square_size,
        (target_location +.5) * pix_square_size,
        width=SHOT_WIDTH,
        )
    
def render_objective(canvas, objective_location, objective_radius, pix_square_size):
    pygame.draw.circle(
        canvas,
        (255, 0, 0),
        (objective_location + 0.5) * pix_square_size,
        pix_square_size * objective_radius,
    )


def render_model(canvas, colour, location, pix_square_size):
    pygame.draw.circle(
            canvas,
            colour,
            (location + 0.5) * pix_square_size,
            pix_square_size / 2,
        )

def render_grid(self, canvas, pix_square_size):
    for x in range(self.size + 1):
        pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=GRID_WIDTH,
            )
        pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=GRID_WIDTH,
            )

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([qx, qy])