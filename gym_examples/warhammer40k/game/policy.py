import numpy as np

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        # register keyboard events with this environment's window
        # env.viewers[agent_index].window.on_key_press = self.key_press
        # env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        return np.empty(2)

    # keyboard event callbacks
    def key_press(self, k, mod):
    #     if k==key.LEFT:  self.move[0] = True
    #     if k==key.RIGHT: self.move[1] = True
    #     if k==key.UP:    self.move[2] = True
    #     if k==key.DOWN:  self.move[3] = True
        
    # def key_release(self, k, mod):
    #     if k==key.LEFT:  self.move[0] = False
    #     if k==key.RIGHT: self.move[1] = False
    #     if k==key.UP:    self.move[2] = False
    #     if k==key.DOWN:  self.move[3] = False
        pass