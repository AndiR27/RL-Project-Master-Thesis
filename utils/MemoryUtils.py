import numpy as np
from collections import deque


# ReplayBuffer class : Pour DQN et SAC
class MemoryReplay:
    def __init__(self, capacity, seed=1):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        np.random.seed(seed)

    def store(self, state, action, reward, done, next_state):
        transition = (np.asarray(state, dtype=np.float32),
                      np.asarray(action, dtype=np.int64),
                      np.asarray(reward, dtype=np.float32),
                      np.asarray(done, dtype=np.float32),
                      np.asarray(next_state, dtype=np.float32))

        self.memory.append(transition)

    def sample(self, batch_size):
        # Récupère un échantillon aléatoire de la mémoire selon la taille du batch
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]

        state, action, reward, done, next_state = zip(*batch)

        return (np.array(state),
                np.array(action),
                np.array(reward),
                np.array(done),
                np.array(next_state))

    def __len__(self):
        return len(self.memory)

    def get_memory_size(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()


# Gestion de la memoire pour PPO
class MemoryPPO:
    def __init__(self, max_size):
        self.max_size = max_size
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)
        self.log_probs = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def sample_all(self):
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        log_probs = np.array(self.log_probs)
        values = np.array(self.values)

        return states, actions, rewards, dones, log_probs, values

    def get_memory_size(self):
        return len(self.states)

#
