import wandb
from algorithms.DQN import DQN_Agent
from algorithms.PPO import PPO_Agent
from algorithms.SAC import SAC_Agent
from environments.GestionEnv import EnvironnementGym

from environments.Wrapper import StartPositionWrapper
from training.Config_hyperparameters import SAC_CONFIG, SAC_CONFIG_BREAKOUT, PPO_CONFIG_BREAKOUT, DQN_CONFIG_BREAKOUT


# Classe pour l'entraînement de l'agent sur Breakout :
# Quelques différences avec CartPole : on a un nombre d'épisodes plus grand, un nombre de pas plus grand, et on a une
# fonction lancementJeu qui prend en compte le fait que l'on peut perdre des vies.
# De plus, un wrapper est utilisé pour définir les positions de départ.
class TrainerBreakout:
    def __init__(self, agent, env, num_episodes, max_steps, seed):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.seed = seed
        self.step = 0

    def lancementJeu(self, eval_mode=False):
        # Initialisation du stockage, de l'environnement et démarrage du compteur de récompense :

        state, info = self.env.reset()
        episode_reward = 0
        for step in range(self.max_steps):
            # le jeu a commencé, on effectue une action : on donne l'état actuel et l'indice de l'étape
            action = self.agent.choose_action(state, eval_mode=eval_mode)
            # on fait un pas dans l'environnement et on récupère le next state, la récompense, et si done ou pas (le reste sont des infos)
            next_state, reward, done, _, info = self.env.step(action)
            self.step += 1
            # Stocker l'expérience dans la mémoire
            self.agent.storeXP(state, action, reward, done, next_state)
            # L'état change + on a une récompense
            state = next_state
            episode_reward += reward
            self.agent.train()

            # si done , on stop l'episode en cours
            # Dans certaines expériences, la perte d'une vie entraine un done. (Facultatif)
            if done and info['lives'] == 0:
                break

        # on retourne la récompense
        return episode_reward

    def train(self):
        # create wrapper
        self.env = StartPositionWrapper(self.env)
        self.agent.memory.clear()
        list_rewards = []

        for episode in range(self.num_episodes):
            episode_recompense_totale = self.lancementJeu()
            list_rewards.append(episode_recompense_totale)
            wandb.log({"Episode": episode, "Reward": episode_recompense_totale})
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {episode_recompense_totale}")

            # self.agent.infos(episode)


if __name__ == "__main__":
    run = wandb.init(project="dqn_breakout", config=DQN_CONFIG_BREAKOUT)
    # agent_params = run_config[agent_name]
    agent_params = DQN_CONFIG_BREAKOUT
    config = wandb.config
    env = EnvironnementGym('Breakout').choix_env()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = DQN_Agent(input_dim, output_dim, **agent_params)

    trainer = TrainerBreakout(agent, env, num_episodes=100000, max_steps=100_000, seed=config.seed)
    trainer.train()
    run.finish()
