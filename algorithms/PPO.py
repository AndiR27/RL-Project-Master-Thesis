import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import torch.nn.utils as nn_utils
#importer la classe PPOMemory de memory.py
from utils.memory import Memory


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
            nn.Linear(hidden_dim, hidden_dim),  # Troisième couche linéaire avec hidden_dim neurones
            nn.Tanh(),  # Activation Tanh
            nn.Linear(hidden_dim, output_dim)  # Dernière couche linéaire avec la dimension de sortie
        )

        # Optimiseur Adam avec un taux d'apprentissage de learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Configuration de l'appareil (GPU ou CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
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
            nn.Linear(hidden_dim, hidden_dim),  # Troisième couche linéaire avec hidden_dim neurones
            nn.Tanh(),  # Activation Tanh
            nn.Linear(hidden_dim, 1)  # Dernière couche linéaire avec une seule sortie (valeur d'état)
        )

        # Optimiseur Adam avec un taux d'apprentissage de learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Configuration de l'appareil (GPU ou CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        self.to(self.device)

    def forward(self, x):
        # Passage en avant (forward pass)
        value = self.critic(x)
        return value


# La classe PPO encapsule toute la logique de l'agent dans l'algorithme Proximal Policy Optimization.
# Elle gère les réseaux d'acteur et de critique, prend des décisions, collecte des expériences, et met à jour les politiques basées sur ces expériences.
class PPOAgent:
    def __init__(self, actor_input_dim, actor_output_dim, critic_input_dim,
                 actor_hidden_dim, critic_hidden_dim, actor_lr, critic_lr,
                 gamma, lambda_advantage, clip_epsilon, entropy_coeff, epochs, batch_size):
        self.actor = PPOActorNetwork(actor_input_dim, actor_output_dim, actor_hidden_dim, actor_lr)
        self.critic = PPOCriticNetwork(critic_input_dim, critic_hidden_dim, critic_lr)
        self.memory = Memory(fields=["state", "action", "reward", "done", "log_prob", "value"])
        self.gamma = gamma
        self.lambda_advantage = lambda_advantage
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_actor = []
        self.loss_critic = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Utilisée lors de l'interaction de l'agent avec l'environnement pour choisir des actions en fonction des états actuels.
    def choose_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0).to(self.actor.device)
        dist = self.actor.get_distribution(state)  # Passe l'état à travers le réseau d'acteur pour obtenir une distribution de probabilité des actions.
        action = dist.sample()  # Échantillonne une action à partir de la distribution.

        return action.item()   # Retourne l'action

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

    def compute_returns(self, rewards):
        """
        Return discounted rewards based on the given rewards and gamma param.
        """
        # Assurez-vous que rewards est un numpy array
        rewards = np.array(rewards, dtype=np.float32)

        # Initialiser un tableau pour les nouvelles récompenses avec la même taille que rewards
        discounted_rewards = np.zeros_like(rewards)

        # La dernière récompense actualisée est identique à la dernière récompense
        discounted_rewards[-1] = rewards[-1]

        # Calcul des récompenses actualisées en itérant à rebours
        for t in reversed(range(len(rewards) - 1)):
            discounted_rewards[t] = rewards[t] + self.gamma * discounted_rewards[t + 1]

        return discounted_rewards

    def normalizeAdvantages(self, tensor):
        mean = tensor.mean()
        std = tensor.std() + 1e-8
        return (tensor - mean) / std

    def trainActor(self, states, actions, old_log_probs, advantages):
        actor_loss_total = 0
        for _ in range(self.epochs):
            self.actor.optimizer.zero_grad()

            # Calculer la perte d'acteur en utilisant les observations (obs), actions (acts),
            # log_probs précédents (old_log_probs), et GAEs
            dist = self.actor.get_distribution(states)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # gestion avec le clip
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            # Perte d'acteur (Objective PPO clipped)

            # Calcul de l'entropie pour régularisation (facultatif)
            entropy = dist.entropy().mean()
            actor_loss = (-torch.min(surr1, surr2) - self.entropy_coeff * entropy).mean()

            # Optimiser la politique
            actor_loss.backward()
            nn_utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor.optimizer.step()
            actor_loss_total += actor_loss.item()
            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= 0.01:
                break

        self.loss_actor.append(actor_loss_total / self.epochs)

    def trainCritic(self, states, returns):
        critic_loss_total = 0
        for _ in range(self.epochs):
            self.critic.optimizer.zero_grad()

            # Calculer les valeurs d'état prédites
            values = self.critic(states).squeeze()

            # Calculer la perte du critique (MSE entre les valeurs prédites et les retours)
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()
            # Optimiser le réseau critique
            value_loss.backward()
            nn_utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic.optimizer.step()
            critic_loss_total += value_loss.item()

        self.loss_critic.append(critic_loss_total / self.epochs)

    def train(self):
        # Gestion des données
        states, actions, old_log_probs, advantages, returns = self.gestionDonnees()

        # Entraînement de l'acteur et du critique
        self.trainActor(states, actions, old_log_probs, advantages)
        self.trainCritic(states, returns)
        self.memory.clear()



    def getLoses(self, episode):
        # Affiche la loss de l'actor et du critic selon l'episode en cours stocké à partir des list loss_actor et loss_critic
        if episode < len(self.loss_actor) and episode < len(self.loss_critic):
            print(f"Episode {episode}, Actor Loss: {self.loss_actor[episode]}, Critic Loss: {self.loss_critic[episode]}")
        else:
            print(f"Episode {episode} is out of range for recorded losses.")

    def gestionDonnees(self):
        device_tensor = self.actor.device
        # Convertir les expériences en numpy arrays
        #self.memory = Memory(fields=["state", "action", "reward", "done", "log_prob", "value"])
        states, actions, rewards, dones, log_probs, values = self.memory.sample_all().values()

        # Calculer les avantages et les retours
        advantages = self.compute_advantages(rewards, values)
        advantages = self.normalizeAdvantages(advantages)
        returns = self.compute_returns(rewards)

        # Mélanger les données
        permute_idxs = np.random.permutation(len(states))
        states = states[permute_idxs]
        actions = actions[permute_idxs]
        log_probs = log_probs[permute_idxs]
        advantages = advantages[permute_idxs]
        returns = returns[permute_idxs]

        # Transformer en tensors
        device_tensor = self.actor.device
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device_tensor)
        actions_tensor = torch.tensor(actions, dtype=torch.int32, device=device_tensor)
        log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32, device=device_tensor)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device_tensor)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device_tensor)

        return states_tensor, actions_tensor, log_probs_tensor, advantages_tensor, returns_tensor

    def storeXP(self, state, action, reward, done, next_state=None):
        #Stocker l'expérience dans la mémoire
        #mais avant, pour PPO on a besoin de calculer la log_prob et la valeur de l'état (pas de tensors)
        state_np = np.array(state)
        state_tensor = torch.tensor(state_np, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor([action]).to(self.device)

        # Calculer la distribution et les valeurs en utilisant le modèle d'acteur
        dist = self.actor.get_distribution(state_tensor)
        value = self.critic(state_tensor)

        log_prob = dist.log_prob(action_tensor).item()
        value = value.item()


        self.memory.store(state=state, action=action, reward=reward, done=done, log_prob=log_prob, value=value)




        """
        state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0).to(self.actor.device)
        dist = self.actor.get_distribution(state)
        action_tensor = torch.tensor([action], dtype=torch.int64).to(self.actor.device)
        log_prob = dist.log_prob(action_tensor)  # Calcule la probabilité logarithmique de l'action échantillonnée.
        value = self.critic(state)  # Passe l'état à traver le réseau critic pour obtenir la valeur estimée
        """