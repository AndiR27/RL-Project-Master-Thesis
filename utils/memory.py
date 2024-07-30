import numpy as np


class Memory:
    def __init__(self, fields, buffer_size=10000):
        self.fields = fields
        self.buffer_size = buffer_size
        self.data = {field: [] for field in self.fields}

    def store(self, **kwargs):
        for field, value in kwargs.items():
            if field in self.data:
                self.data[field].append(value)
                # Remove oldest experience if buffer is full
                if len(self.data[field]) > self.buffer_size:
                    self.data[field].pop(0)

    def sample(self, batch_size):
        memory_size = self.get_memory_size()
        if memory_size < batch_size:
            batch_size = memory_size
        indices = np.random.choice(memory_size, batch_size, replace=False)
        return {field: np.array([self.data[field][i] for i in indices]) for field in self.data}

    def sample_all(self):
        return {field: np.array(values) for field, values in self.data.items()}

    def clear(self):
        self.data = {field: [] for field in self.data}

    def shuffle(self):
        permute_idxs = np.random.permutation(len(self.data[next(iter(self.data))]))
        for field in self.data:
            self.data[field] = [self.data[field][i] for i in permute_idxs]

    def get_memory_size(self):
        return len(self.data[self.fields[0]])
