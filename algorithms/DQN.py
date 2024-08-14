import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from utils.MemoryUtils import MemoryReplay


# Neural Network Model
class DQN_Network(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=256, learning_rate=0.0001):
        super(DQN_Network, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=50000, gamma=0.9)

        self.loss_fn = nn.MSELoss()
        # Configuration de l'appareil (GPU ou CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

    def update_learning_rate(self):
        self.scheduler.step()


class DQN_Agent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, learning_rate=0.0001, gamma=0.99, batch_size=128,
                 replay_memory_size=500_000, min_replay_memory_size=1000,
                 update_target_every=1000, epsilon_start=1, epsilon_decay=1000, epsilon_min=0.001, seed=1):
        # Fixer la seed pour la reproductibilité
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Model principal => Initialisé de façon aléatoire et l'agent prendra des actions aléatoires => mais sera entrainé à chaque étape
        self.model = DQN_Network(state_dim, action_dim, hidden_dim, learning_rate)

        self.n_actions = action_dim
        # Target model => pour avoir une certaine consistence dans la prédiction,
        # on a ce model qui est prédit à chaque étape => Va permettre au model d'apprendre
        # les poids du target model sont mis à jour avec les poids du main model à chaque update_target_every étapes
        self.target_model = DQN_Network(state_dim, action_dim, hidden_dim, learning_rate)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # replay memory : va contenir <S, A, R, done, S'> => avec deux modeles,
        self.memory = MemoryReplay(replay_memory_size, seed=seed)

        # Permet de compter quant il faut mettre à jour le model target avec les poids du main network
        self.target_update_counter = 0
        self.min_replay_memory_size = min_replay_memory_size
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.stepDone = 0
        self.loss = []

    """
    Méthode pour afficher les informations de l'agent (optionnel)
    @return: string
    """

    def infos(self):
        infos = {
            "Loss": np.mean(self.loss) if self.loss else 0,
        }
        return infos
    # Ajoute les données des étapes dans le memory replay
    # On y retrouve : S, A, R, done, S'
    def storeXP(self, state, action, reward, done, next_state):
        self.memory.store(state=state, action=action, reward=reward, done=done, next_state=next_state)

    # Main network est utilisé pour évaluer et fournir des estimations de la qualité (valeurs Q) des actions possibles,
    # en se basant sur l'état actuel de l'environnement observé.
    def choose_action(self, state, eval_mode=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Greedy policy
        if eval_mode or np.random.rand() > self.epsilon:
            with torch.no_grad():
                q_values = self.model(state)
                action = torch.argmax(q_values, dim=1).item()
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def update_epsilon(self):

        self.stepDone += 1
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * math.exp(
            -self.epsilon_decay * self.stepDone)
        # wandb.log({"Epsilon": self.epsilon})


    def train(self):
        # Vérifie si la taille de la mémoire est suffisante pour l'entraînement
        if self.memory.get_memory_size() < self.min_replay_memory_size:
            return
        # Échantillonne un lot d'expériences de la mémoire
        states, actions, rewards, dones, next_states = self.memory.sample(self.batch_size)

        # Convertit les expériences échantillonnées en tenseurs PyTorch et les transfère sur l'appareil spécifié (CPU ou GPU)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)

        # Calcule les Q-valeurs actuelles pour les actions prises dans les états échantillonnés
        current_qs = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Calcule les Q-valeurs des prochains états en utilisant le réseau cible
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)  # Les meilleures actions dans les prochains états
            next_qs = self.target_model(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)

        # Assure que les 'dones' sont des booléens et les convertit en float pour les opérations arithmétiques
        dones = dones.float()
        # Calcule les Q-valeurs cibles pour les états actuels
        target_qs = rewards + (self.gamma * next_qs * (1 - dones))

        # Calcule la perte entre les Q-valeurs actuelles et les Q-valeurs cibles
        loss = self.model.loss_fn(current_qs, target_qs)

        # Effectue la rétropropagation et optimise les poids du réseau
        self.model.optimizer.zero_grad()
        loss.backward()

        # Clip les gradients en place pour stabiliser l'entraînement
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.model.optimizer.step()
        self.model.update_learning_rate()

        # Journalise la valeur de la perte pour le suivi et la visualisation
        # wandb.log({"Loss": loss.item()})
        self.loss.append(loss.item())

        # Met à jour la valeur epsilon pour la politique epsilon-greedy
        self.update_epsilon()

        # Met à jour le réseau cible à chaque 'update_target_every' étapes
        self.update_target_network()

        # Mise à jour douce du réseau cible pour maintenir la stabilité
        # self.soft_update(self.target_model, self.model, tau=0.005)

    def update_target_network(self):
        self.target_update_counter += 1
        if self.target_update_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0


    def soft_update(self, target, source, tau=0.001):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
