import gymnasium as gym


# gestion avec une classe qui va retourner l'environnement
class EnvironnementGym:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = None

    def create_cartpole_env(self):
        env = gym.make('CartPole-v1')
        self.env = env
        return env

    def create_breakout_env(self):
        env = gym.make('ALE/Breakout-ram-v5')
        self.env = env
        return env


    def choix_env(self):
        if self.env_name == 'CartPole-v1':
            return self.create_cartpole_env()
        elif self.env_name == 'Breakout':
            return self.create_breakout_env()
        else:
            raise ValueError("Environnement non reconnu")

    def state_action_space(self):
        return self.env.observation_space.shape[0], self.env.action_space.n
