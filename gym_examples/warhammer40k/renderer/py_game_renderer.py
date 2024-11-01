from gym_examples.warhammer40k.game.world import World
import pygame

class PyGameRenderer:
    
    def __init__(self):
        self.grid_width = 1
        self.window = None
        self.clock = None
        self.canvas = None
        self.window_size = None
        self.pix_square_size = None
        self.game_size = None
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    
    def setup(self, window_size, game_size):
        pygame.init()
        pygame.display.init()
        
        self.window_size = window_size
        self.game_size = game_size
        
        self.window = pygame.display.set_mode((window_size, window_size))
        
        self.clock = pygame.time.Clock()
        
        self.canvas = pygame.Surface((window_size, window_size))
        self.canvas.fill((255, 255, 255))
        self.pix_square_size = (
            self.window_size / game_size
        )  # The size of a single grid square in pixels
        
        
    def render_frame(self, world: World):
        
        
        self.render_grid()
        
        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
    
    def render_grid(self):
        for x in range(self.game_size + 1):
            pygame.draw.line(
                    self.canvas,
                    0,
                    (0, self.pix_square_size * x),
                    (self.window_size, self.pix_square_size * x),
                    width=self.grid_width,
                )
            pygame.draw.line(
                    self.canvas,
                    0,
                    (self.pix_square_size * x, 0),
                    (self.pix_square_size * x, self.window_size),
                    width=self.grid_width,
                )
