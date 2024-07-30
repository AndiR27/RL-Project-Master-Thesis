from collections import deque
from random import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import torch.nn.utils as nn_utils
from utils.memory import Memory  # Importer la classe Memory de memory.py


# Neural Network Model
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=256, learning_rate=0.0001):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Ajout d'une couche supplémentaire
        self.out = nn.Linear(hidden_dim, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        # Configuration de l'appareil (GPU ou CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)


class DQNAgent:
    def __init__(self, input_shape, n_actions, hidden_dim=256, learning_rate=0.0001, gamma=0.99, batch_size=256,
                 replay_memory_size=50_000, min_replay_memory_size=1000,
                 update_target_every=5, epsilon=1.0, epsilon_decay=0.95, epsilon_min=0.01):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Model principal => Initialisé de façon aléatoire et l'agent prendra des actions aléatoires => mais sera entrainé à chaque étape
        self.model = DQN(input_shape, n_actions, hidden_dim, learning_rate)
        self.n_actions = n_actions
        # Target model => pour avoir une certaine consistence dans la prédiction, on a ce model qui est prédit à chaque étape => Va permettre au model d'apprendre
        self.target_model = DQN(input_shape, n_actions, hidden_dim, learning_rate)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # replay memory : va contenir <S, A, R, S'> => avec deux modeles, ça facilite le problème de stocker les infos dans cette liste (taille de la liste = nombre d'étapes)
        self.memory = Memory(fields=["state", "action", "reward", "done", "next_state"], buffer_size=replay_memory_size)

        # Permet de compter quant il faut mettre à jour le model target avec les poids du main network
        self.target_update_counter = 0
        self.min_replay_memory_size = min_replay_memory_size
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    # Ajoute les données des étapes à la liste de memory replay
    # On y retrouve : S, A, R, S'
    def storeXP(self, state, action, reward, done, next_state):
        self.memory.store(state=state, action=action, reward=reward, done=done, next_state=next_state)
        #print(f"Stored experience - Memory size: {self.memory.get_memory_size()}")

    # Methode pour récupérer les Q-values
    # Interroge le main network pour obtenir les valeurs Q étant donné l'espace d'observation actuel (état de l'environnement)
    # Main network est utilisé pour évaluer et fournir des estimations de la qualité (valeurs Q) des actions possibles, en se basant sur l'état actuel de l'environnement observé.
    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array(state)).float().unsqueeze(0).to(self.device)
            q_values = self.model(state)
            action = torch.argmax(q_values, dim=1).item()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def train(self):
        if self.memory.get_memory_size() < self.min_replay_memory_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, dones, next_states = batch.values()
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)

        current_qs = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_qs = self.target_model(next_states).max(1)[0]
        target_qs = rewards + self.gamma * next_qs * (~dones)


        loss = self.model.loss_fn(current_qs, target_qs)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

        self.target_update_counter += 1
        if self.target_update_counter % self.update_target_every == 0:
            self.update_target_network()
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
