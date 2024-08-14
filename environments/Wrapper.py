import gymnasium as gym
import numpy as np


# Classe pour gérer l'environnement : permet de repositionner le début du jeu à d'autres positions
class StartPositionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(StartPositionWrapper, self).__init__(env)
        self.start_positions = [
            [(0, 10), (1, 20), (2, 30)],  # Example start position 1
            [(0, 40), (1, 50), (2, 60)],  # Example start position 2
            [(0, 70), (1, 80), (2, 90)],  # Example start position 3
        ]

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        start_position = self.start_positions[np.random.choice(len(self.start_positions))]
        # Apply each (position, value) pair to the RAM
        for position, value in start_position:
            self.env.unwrapped.ale.setRAM(position, value)
        return state, info

    def step(self, action):
        return self.env.step(action)

    def getStatesSize(self):
        return self.env.observation_space.shape[0]

    def getActionsSize(self):
        return self.env.action_space.n

    def __getattr__(self, name):
        return getattr(self.env, name)
