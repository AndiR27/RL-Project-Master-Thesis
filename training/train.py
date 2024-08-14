# Classe pour gérer l'entraînement
import numpy as np
import wandb


# Classe permettant de gérer l'entraînement de l'agent sur l'environnement
# @param agent: l'agent à entraîner
# @param env: l'environnement sur lequel l'agent est entraîné
# @param num_episodes: nombre d'épisodes d'entraînement
# @param max_steps: nombre maximal de pas par épisode
# @param num_eval_episodes: nombre d'épisodes d'évaluation
# @param seed: seed pour la reproductibilité
class Trainer:
    def __init__(self, agent, env, num_episodes, max_steps, num_eval_episodes, seed):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.num_eval_episodes = num_eval_episodes
        self.seed = seed
        self.seed_increment = 0
        self.step = 0
        self.step_eval = 0

    ## Fonction qui lance le jeu et retourne la récompense totale
    def lancementJeu(self, eval_mode=False):
        # Initialisation de l'environnement et démarrage du compteur de récompense :
        state, _ = self.env.reset(seed=self.seed + self.seed_increment)
        self.seed_increment += 10
        episode_reward = 0
        self.step_eval = 0

        for i in range(self.max_steps):
            # le jeu a commencé, on effectue une action : on donne l'état actuel
            action = self.agent.choose_action(state, eval_mode=eval_mode)

            # on fait un pas dans l'environnement et on récupère le next state, la récompense, et si done ou pas (le reste sont des infos)
            next_state, reward, done, _, info = self.env.step(action)

            if eval_mode:
                self.step_eval += 1  # Increment eval_step during evaluation
            else:
                self.step += 1  # Increment training step during training

            # Stocker l'expérience dans la mémoire
            self.agent.storeXP(state, action, reward, done, next_state)
            # L'état change + on a une récompense
            state = next_state
            episode_reward += reward

            # Entraînement de l'agent
            if (
                    self.agent.__class__.__name__ == "DQN_Agent" or self.agent.__class__.__name__ == "SAC_Agent") and not eval_mode:
                self.agent.train()
            # si done = on est tombé, on stop l'episode en cours
            if done:
                break

        # on retourne la récompense
        return episode_reward, self.step_eval

    # Fonction principale pour l'entraînement de l'agent
    def train(self, logging_prints=False, no_eval=False):
        # liste des recompenses par episodes:
        # on vide la mémoire (au cas où)
        self.agent.memory.clear()
        list_rewards = []

        for episode in range(1, self.num_episodes + 1):

            # on lance le jeu et on récupère le Return (récompense totale)
            episode_recompense_totale, _ = self.lancementJeu()

            # on stocke la récompense totale
            list_rewards.append(episode_recompense_totale)

            # Dans le cas de PPO, on entraîne l'agent à la fin de chaque épisode
            if self.agent.__class__.__name__ == "PPO_Agent":
                self.agent.train()

            if logging_prints:
                print(f"Episode {episode}, Total Reward: {episode_recompense_totale}")
                for key, value in self.agent.infos().items():
                    print(f"{key}: {value}")
            else:
                wandb.log({"Train/Episode": episode, "Train/Reward": episode_recompense_totale})
                for key, value in self.agent.infos().items():
                    wandb.log({f"Train/{key}": value})

            # Affichage de la récompense totale tous les 40 épisodes + évaluation
            if episode % 40 == 0 and not no_eval:
                print(f"Episode {episode}, Total Reward: {episode_recompense_totale}")

                # La il faudrait evaluer l'agent
                list_rewards_eval, list_steps_eval = self.evaluate()
                # Log evaluation stats
                wandb.log({
                    "Eval/mean_reward": np.mean(list_rewards_eval),
                    "Eval/mean_steps": np.mean(list_steps_eval),
                    "episode": episode,
                    "phase": "evaluation"
                })

        if logging_prints:
            print("Somme des récompenses par épisode: ", (sum(list_rewards) / self.num_episodes))
        else:
            wandb.log({"Total Reward": sum(list_rewards) / self.num_episodes})

    # Fonction pour évaluer l'agent : on lance le jeu en mode évaluation
    # Pas d'entraînement de l'agent, les actions sont choisies en fonction de la politique apprise
    def evaluate(self):
        list_rewards = []
        list_steps = []
        for episode in range(self.num_eval_episodes):
            episode_recompense_totale, eval_step = self.lancementJeu(eval_mode=True)
            list_rewards.append(episode_recompense_totale)
            list_steps.append(eval_step)

        return list_rewards, list_steps
