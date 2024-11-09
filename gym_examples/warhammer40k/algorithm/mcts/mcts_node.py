import numpy as np

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Store the action leading to this node
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self, action_space):
        return len(self.children) == len(action_space)

    def best_child(self, exploration_weight=1.0):
        # UCT formula: exploitation + exploration
        return max(
            self.children,
            key=lambda child: child.value / (child.visits + 1e-6) + 
            exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
        )
