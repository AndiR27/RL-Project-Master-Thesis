# Classe pour gérer l'entraînement
class Trainer:
    def __init__(self, agent, env, num_episodes, max_steps):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    ## Fonction qui lance le jeu et retourne la récompense
    def lancementJeu(self):
        # Initialisation du stockage, de l'environnement et démarrage du compteur de récompense :
        # ppo_agent.memory.sample()

        state, _ = self.env.reset()
        episode_reward = 0
        for _ in range(self.max_steps):
            # le jeu a commencé, on effectue une action
            action = self.agent.choose_action(state)
            # on fait un pas dans l'environnement et on récupère le next state, la récompense, et si done ou pas (le reste sont des infos)
            next_state, reward, done, _, _ = self.env.step(action)
            # Stocker l'expérience dans la mémoire
            self.agent.storeXP(state, action, reward, done, next_state)
            # L'état change + on a une récompense
            state = next_state
            episode_reward += reward
            # si done = on est tombé, on stop l'episode en cours
            if done:
                break

        # on retourne la récompense
        return episode_reward

    def train(self):
        # liste des recompenses par episodes:
        list_rewards = []
        for episode in range(self.num_episodes):
            # on lance le jeu et on récupère la récompense
            episode_recompense_totale = self.lancementJeu()
            # on stocke la récompense
            list_rewards.append(episode_recompense_totale)
            # on fait un pas dans l'optimisation
            self.agent.train()
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {episode_recompense_totale}")

        print("Somme des récompenses par épisode: ", (sum(list_rewards)/self.num_episodes))

