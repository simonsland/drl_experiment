import numpy as np


class Random:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def choose_action(self, observation):
        return np.random.choice(np.arange(self.n_actions), 1)[0]
