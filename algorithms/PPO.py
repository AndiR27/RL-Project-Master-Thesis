import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import torch.nn.utils as nn_utils
from utils.MemoryUtils import MemoryPPO


# ActorNetwork : Réseau de neurones pour l'acteur, qui génère des actions basées sur les états observés.
# Rôle : Le réseau d'acteur (ActorNetwork) est responsable de choisir les actions à prendre dans l'environnement en fonction des états observés.
# Il génère une distribution de probabilités sur les actions possibles et sélectionne les actions à partir de cette distribution.
# Pourquoi ? : L'acteur est essentiel pour la prise de décision. En PPO, l'objectif est de maximiser une fonction de récompense
# en ajustant la politique de l'acteur pour choisir des actions qui mènent à des récompenses plus élevées.
class PPOActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate):
        super(PPOActorNetwork, self).__init__()

        # Définition de l'architecture du réseau d'acteur
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Première couche linéaire avec hidden_dim neurones
            nn.Tanh(),  # Activation Tanh
            nn.Linear(hidden_dim, hidden_dim),  # Deuxième couche linéaire avec hidden_dim neurones
            nn.Tanh(),  # Activation Tanh
            nn.Linear(hidden_dim, output_dim)  # Dernière couche linéaire avec la dimension de sortie
        )

        # Configuration de l'appareil (GPU ou CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # Passage en avant (forward pass)
        return self.actor(x)

    def get_distribution(self, x):
        logits = self.actor(x)
        return Categorical(logits=logits)


# Réseau de neurones pour le critique, qui évalue les états et aide à calculer les avantages.
# Rôle : Le réseau critique (CriticNetwork) évalue les états en estimant la valeur attendue (valeur d'état).
# Cette estimation est utilisée pour calculer l'avantage (advantage) des actions prises, qui est une mesure de combien une action est meilleure que la moyenne attendue.
# Pourquoi ? : Le critique est essentiel pour fournir un signal d'apprentissage au réseau d'acteur.
# En PPO, l'avantage est utilisé pour mettre à jour la politique de l'acteur de manière plus stable et efficace.
# Le critique aide à réduire la variance des estimations de retour en fournissant des évaluations plus précises des états.
class PPOCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, learning_rate):
        super(PPOCriticNetwork, self).__init__()
        # Définition de l'architecture du réseau critique
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Première couche linéaire avec hidden_dim neurones
            nn.Tanh(),  # Activation Tanh
            nn.Linear(hidden_dim, hidden_dim),  # Deuxième couche linéaire avec hidden_dim neurones
            nn.Tanh(),  # Activation Tanh
            nn.Linear(hidden_dim, 1)  # Dernière couche linéaire avec une seule sortie (valeur d'état)
        )

        # Configuration de l'appareil (GPU ou CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # Passage en avant (forward pass)
        value = self.critic(x)
        return value


# La classe PPO encapsule toute la logique de l'agent dans l'algorithme Proximal Policy Optimization.
# Elle gère les réseaux d'acteur et de critique, prend des décisions, collecte des expériences, et met à jour les politiques basées sur ces expériences.
class PPO_Agent:
    def __init__(self, state_dim, action_dim, actor_hidden_dim, critic_hidden_dim, actor_lr, critic_lr,
                 gamma, lambda_advantage, clip_epsilon, entropy_coeff, epochs, batch_size, seed):

        # Fixer la seed pour la reproductibilité
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

        # Networks
        self.actor = PPOActorNetwork(state_dim, action_dim, actor_hidden_dim, actor_lr)
        self.critic = PPOCriticNetwork(state_dim, critic_hidden_dim, critic_lr)
        # Optimizer commun pour les réseaux d'acteur et de critique
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=actor_lr,
                                    eps=1e-5)

        # Gestion de la memoire
        self.memory = MemoryPPO(100000)

        self.gamma = gamma
        self.lambda_advantage = lambda_advantage
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_actor_list = []
        self.loss_critic_list = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Utilisée lors de l'interaction de l'agent avec l'environnement pour choisir des actions en fonction des états actuels.
    def choose_action(self, state, eval_mode=False):
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0).to(self.actor.device)
        # Passe l'état à travers le réseau d'acteur pour obtenir une distribution de probabilité des actions.
        dist = self.actor.get_distribution(state)
        if eval_mode:
            action = dist.probs.argmax().item()
            return action

        action = dist.sample()  # Échantillonne une action à partir de la distribution.
        return action.item()  # Retourne l'action

    def normalize_observation(self, obs):
        obs_mean = np.mean(obs)
        obs_std = np.std(obs) + 1e-8
        normalized_obs = (obs - obs_mean) / obs_std
        clipped_obs = np.clip(normalized_obs, -10, 10)
        return clipped_obs

    # Calcul des avantages en utilisant les récompenses et les valeurs prédites par le réseau critique.
    # Les avantages sont utilisés pour mettre à jour la politique de l'acteur de manière plus stable et efficace.
    # Les avantages sont calculés en utilisant la méthode Generalized Advantage Estimation (GAE).
    def compute_advantages(self, rewards, values):
        # Calcul des next values
        next_values = np.append(values[1:], 0)

        # Calcul des deltas
        deltas = rewards + self.gamma * next_values - values

        # Calcul des GAEs
        gaes = np.zeros_like(deltas)
        gaes[-1] = deltas[-1]
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = deltas[t] + self.gamma * self.lambda_advantage * gaes[t + 1]

        return gaes

    # Calcul des retours actualisés en utilisant les récompenses reçues par l'agent.
    # Les retours sont utilisés pour mettre à jour le réseau critique.
    def compute_returns(self, rewards):
        rewards = np.array(rewards, dtype=np.float32)

        # Initialiser un tableau pour les nouvelles récompenses avec la même taille que rewards
        discounted_rewards = np.zeros_like(rewards)

        # La dernière récompense actualisée est identique à la dernière récompense
        discounted_rewards[-1] = rewards[-1]

        # Calcul des récompenses actualisées en itérant à rebours
        for t in reversed(range(len(rewards) - 1)):
            discounted_rewards[t] = rewards[t] + self.gamma * discounted_rewards[t + 1]

        # Normalize discounted rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (
                np.std(discounted_rewards) + 1e-8)

        return discounted_rewards

    def normalizeAdvantages(self, tensor):
        mean = tensor.mean()
        std = tensor.std() + 1e-8
        return (tensor - mean) / std

    # Mwethode permettant de calculer la loss CLIP et la loss de l'entropie
    def trainActor(self, states, actions, old_log_probs, advantages):
        # Réinitialiser le gradient de l'optimiseur de l'acteur
        # self.actor.optimizer.zero_grad()

        # Calculer la perte d'acteur en utilisant les observations (obs), actions (acts),
        # log_probs précédents (old_log_probs), et GAEs
        dist = self.actor.get_distribution(states)
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # clip : permet de limiter le ratio pour éviter des mises à jour trop importantes
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

        # Calcul de l'entropie pour régularisation (facultatif)
        # entropy = torch.mean(torch.sum(-new_logits * (torch.log(new_logits + 1e-5)), dim=1))
        entropy = dist.entropy().mean()

        # Perte d'acteur (Objective PPO clipped)
        actor_loss = (-torch.min(surr1, surr2)).mean()
        return actor_loss, entropy

    # Calcul de la loss du critique
    def trainCritic(self, states, returns):

        # Calculer les valeurs d'état prédites
        values = self.critic(states).squeeze()

        # Calculer la perte du critique (MSE entre les valeurs prédites et les retours)
        value_loss = (values - returns).pow(2).mean()

        return value_loss

    # Entraînement de l'agent PPO en utilisant les expériences stockées dans la mémoire.
    def train(self):
        # Récupérer les données de la mémoire
        states, actions, old_log_probs, advantages, returns = self.gestionDonnees()
        # Calculer le nombre de mini-batchs
        nbatch = len(states)
        nbatch_train = self.batch_size
        inds = np.arange(nbatch)

        # Entraînement de l'agent PPO
        for epoch in range(self.epochs):
            # Mélanger les indices
            np.random.shuffle(inds)

            # Entraîner l'agent pour chaque mini-batch
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]

                mb_states = states[mbinds]
                mb_actions = actions[mbinds]
                mb_old_log_probs = old_log_probs[mbinds]
                mb_advantages = advantages[mbinds]
                mb_returns = returns[mbinds]

                self.optimizer.zero_grad()

                # Récupérer la loss de l'acteur et du critic
                actor_loss, entropy = self.trainActor(mb_states, mb_actions, mb_old_log_probs, mb_advantages)
                critic_loss = self.trainCritic(mb_states, mb_returns)
                self.loss_actor_list.append(actor_loss.item())
                self.loss_critic_list.append(critic_loss.item())

                # Perte totale : LCLIP + VF - EntropyBonus
                loss = actor_loss + critic_loss * 0.5 - self.entropy_coeff * entropy

                loss.backward()

                nn_utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.optimizer.step()

                # Facultatif : vérifier la divergence KL : si elle est trop grande, arrêter l'entraînement
                with torch.no_grad():
                    dist = self.actor.get_distribution(mb_states)
                    new_log_probs = dist.log_prob(mb_actions)
                    kl_div = torch.mean(mb_old_log_probs - new_log_probs)
                    if kl_div > 0.01:
                        break

        self.memory.clear()

    def infos(self):
        infos = {
            "Actor Loss": np.mean(self.loss_actor_list) if self.loss_actor_list else 0,
            "Critic Loss": np.mean(self.loss_critic_list) if self.loss_critic_list else 0
        }
        return infos


    def gestionDonnees(self):
        device_tensor = self.actor.device
        # Convertir les expériences en numpy arrays
        # self.memory = Memory(fields=["state", "action", "reward", "done", "log_prob", "value"])

        states, actions, rewards, dones, log_probs, values = self.memory.sample_all()
        # Calculer les avantages et les retours
        advantages = self.compute_advantages(rewards, values)
        advantages = self.normalizeAdvantages(advantages)
        returns = self.compute_returns(rewards)

        # Transformer en tensors
        device_tensor = self.actor.device

        states_tensor = torch.tensor(states, dtype=torch.float32, device=device_tensor)
        actions_tensor = torch.tensor(actions, dtype=torch.int32, device=device_tensor)
        log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32, device=device_tensor)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device_tensor)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device_tensor)

        return states_tensor, actions_tensor, log_probs_tensor, advantages_tensor, returns_tensor

    def storeXP(self, state, action, reward, done, next_state=None):
        # Stocker l'expérience dans la mémoire
        # mais avant, pour PPO on a besoin de calculer la log_prob et la valeur de l'état (pas de tensors)
        state_np = np.array(state)
        # normalize state
        # state_np = np.clip((state_np - np.mean(state_np) / (np.std(state_np)) + 1e-8), -10, 10)
        state_tensor = torch.tensor(state_np, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor([action]).to(self.device)

        # Calculer la distribution et les valeurs en utilisant le modèle d'acteur
        dist = self.actor.get_distribution(state_tensor)
        value = self.critic(state_tensor)
        log_prob = dist.log_prob(action_tensor).item()
        value = value.item()

        # Stocker l'expérience dans la mémoire
        self.memory.store(state=state, action=action, reward=reward, done=done, log_prob=log_prob, value=value)
