import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from utils.MemoryUtils import MemoryReplay


# Policy Network (Actor)
# Avec SAC : on utilise une politique stochastique pour prendre des actions
class SAC_ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate):
        super(SAC_ActorNetwork, self).__init__()
        # Définition de l'architecture du réseau de politique
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

        # Optimiseur Adam avec un taux d'apprentissage de learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Configuration de l'appareil (GPU ou CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

    def get_action(self, state):
        action_probs = self.forward(state)
        # Gérer la situation des probabilités à 0.0 car on ne peut pas faire log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        action_log_probs = torch.log(action_probs + z)
        return action_probs, action_log_probs


# Critic Network
# Avec SAC : on utilise deux réseaux de valeurs pour estimer les récompenses futures
# Permet d'évaluer la politique et guider la politique vers de meilleures actions
# Evite le problème de maximisation de la fonction de valeur
class SAC_CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate):
        super(SAC_CriticNetwork, self).__init__()

        # Définition de l'architecture du réseau de valeurs
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Optimiseur Adam avec un taux d'apprentissage de learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Configuration de l'appareil (GPU ou CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# SAC Agent
# La classe SAC_Agent est responsable de l'entraînement de l'agent SAC et encapsule les réseaux de politique et de valeurs
# Elle gère également la mémoire et les mises à jour des poids
# Elle est également responsable de la sélection des actions
class SAC_Agent:
    def __init__(self, state_dim, action_dim, hidden_dim, policy_lr, critic_lr, memory_replay_size,
                 min_memory_replay_size, gamma, tau, alpha, batch_size, seed):
        # Fixer la seed pour la reproductibilité
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Actor, Critics et Targets Networks
        self.policy = SAC_ActorNetwork(state_dim, action_dim, hidden_dim, policy_lr).to(self.device)
        self.q1 = SAC_CriticNetwork(state_dim, action_dim, hidden_dim, critic_lr).to(self.device)
        self.q2 = SAC_CriticNetwork(state_dim, action_dim, hidden_dim, critic_lr).to(self.device)
        self.q1_target = SAC_CriticNetwork(state_dim, action_dim, hidden_dim, critic_lr).to(self.device)
        self.q2_target = SAC_CriticNetwork(state_dim, action_dim, hidden_dim, critic_lr).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Replay buffer
        self.memory = MemoryReplay(memory_replay_size, seed=seed)
        self.min_replay_memory_size = min_memory_replay_size

        # Hyperparamètres
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        # Optimiseur pour alpha
        # Grâce à la formule de l'entropie maximale, on peut définir une valeur cible pour l'entropie
        self.target_entropy = 0.1 * -np.log(1 / self.action_dim)

        # Coefficient d'entropie
        self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=5e-4)

        self.loss_actor_list = []
        self.loss_q1_list = []
        self.loss_q2_list = []

    def infos(self):
        infos = {
            "Actor Loss": np.mean(self.loss_actor_list) if self.loss_actor_list else 0,
            "Q1 Loss": np.mean(self.loss_q1_list) if self.loss_q1_list else 0,
            "Q2 Loss": np.mean(self.loss_q2_list) if self.loss_q2_list else 0,
        }
        return infos

    # Choisir une action : Avec SAC, on utilise une politique stochastique pour prendre des actions
    def choose_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, _ = self.policy.get_action(state)

        if eval_mode:
            return torch.argmax(action).item()
        else:
            if self.memory.get_memory_size() < self.min_replay_memory_size:
                return np.random.randint(self.action_dim)
            else:
                action = Categorical(action).sample()
                return action.item()

    # Calculer la loss de la policy et mettre à jour les poids

    def trainActor(self, states):
        # Calculer les probabilités d'actions et les log-probabilités d'actions
        action_probs, log_action_probs = self.policy.get_action(states)

        # Calculer les Q values pour les états actuels
        q1 = self.q1(states)
        q2 = self.q2(states)

        # Récupérer le minimum des deux Q values
        min_Q = torch.min(q1, q2)

        # Récupérer alpha et calculer la loss de la policy
        alpha = self.log_alpha.exp().to(self.device)
        policy_loss = (action_probs * (alpha * log_action_probs - min_Q)).sum(-1).mean()

        # Mettre à jour les poids de la policy
        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy.optimizer.step()
        return policy_loss, log_action_probs

    def update_alpha(self, log_action_probs):
        # Mise à jour de alpha : on utilise la formule de l'entropie maximale
        alpha_loss = - (self.log_alpha.exp().to(self.device) * (log_action_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha)

    """

    """

    # Responsable de l'entraînement des réseaux Q et mettre à jour les valeurs de Q (pour estimer au mieux les recompenses futures de la pair d'état-action)
    # Permet à la fois d'évaluer la policy et guider la policy vers de meilleures actions
    def trainCritic(self, states, actions, rewards, dones, next_states):
        # predire next_state actions et les Q values depuis les target models
        with (torch.no_grad()):
            next_action_probs, next_actions_logs = self.policy.get_action(next_states)  # Next state et pas state
            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_Q = next_action_probs * (torch.min(next_q1, next_q2) - self.alpha.to(self.device) * next_actions_logs)

            # Calculer la target Q pour les états actuels
            # Somme des probabilités d'actions pour produire un tenseur de la forme [batch_size, 1]
            next_Q = next_Q.sum(dim=1, keepdim=True)

            # Calculer la target Q pour les états actuels
            # target_Q = rewards + γ × (1−dones) × next_Q
            target_Q = rewards.float().to(self.device).unsqueeze(1) + (
                    self.gamma * (1 - dones.float().to(self.device)).unsqueeze(1) * next_Q).float()

        actions = actions.unsqueeze(-1)

        # Calculs des loss pour les deux Q networks
        q1 = self.q1(states).gather(1, actions.long())
        q2 = self.q2(states).gather(1, actions.long())

        q1_loss = 0.5 * F.mse_loss(q1, target_Q)
        q2_loss = 0.5 * F.mse_loss(q2, target_Q)

        # Mettre à jour les Q networks
        self.q1.optimizer.zero_grad()
        q1_loss.backward()
        clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1.optimizer.step()

        self.q2.optimizer.zero_grad()
        q2_loss.backward()
        clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2.optimizer.step()

        return q1_loss, q2_loss

    def soft_update_target_networks(self):
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

    # Mise à jour des poids des réseaux cibles
    # On utilise une méthode de soft update pour mettre à jour les poids des réseaux cibles (pour éviter les oscillations)
    # On utilise une valeur de tau pour déterminer la proportion de mise à jour des poids
    def soft_update(self, model, model_target):
        for target_param, local_param in zip(model_target.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    # Entrainement de l'agent
    def train(self):
        # si la taille de la mémoire n'est pas assez grande, on ne peut pas encore entraîner
        if self.memory.get_memory_size() < self.min_replay_memory_size:
            return

        states, actions, rewards, dones, next_states = self.gestionDonnees()

        # Entrainement des différents composants de l'agent
        policy_loss, log_action_probs = self.trainActor(states)
        self.update_alpha(log_action_probs)
        q1_loss, q2_loss = self.trainCritic(states, actions, rewards, dones, next_states)
        self.soft_update_target_networks()
        self.loss_actor_list.append(policy_loss.item())
        self.loss_q1_list.append(q1_loss.item())
        self.loss_q2_list.append(q2_loss.item())

    def gestionDonnees(self):
        # récupérer un sample
        states, actions, rewards, dones, next_states = self.memory.sample(self.batch_size)

        # transformer en tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.int32).to(self.device)
        # dones sont true ou false
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        return states, actions, rewards, dones, next_states

    def storeXP(self, state, action, reward, done, next_state):
        self.memory.store(state=state, action=action, reward=reward, done=done,
                          next_state=next_state)
