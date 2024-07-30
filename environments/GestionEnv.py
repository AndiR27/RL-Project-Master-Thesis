import gymnasium as gym

#gestion avec une classe qui va retourner l'environnement
class Environnement:
    def __init__(self, env_name):
        self.env_name = env_name

    def create_cartpole_env(self):
        env = gym.make('CartPole-v1')
        return env

    def create_breakout_env(self):
        env = gym.make('Breakout-v0')
        return env

    def create_pacman_env(self):
        env = gym.make('Pacman-v0')
        return env

    def choix_env(self):
        if self.env_name == 'CartPole-v1':
            return self.create_cartpole_env()
        elif self.env_name == 'Breakout-v0':
            return self.create_breakout_env()
        elif self.env_name == 'Pacman-v0':
            return self.create_pacman_env()
        else:
            raise ValueError("Environnement non reconnu")



