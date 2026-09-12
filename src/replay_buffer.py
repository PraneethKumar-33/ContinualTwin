import torch
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.x_data = []
        self.y_data = []
        self.items_seen = 0

    def add_data(self, x_new, y_new):
        for i in range(len(x_new)):
            self.items_seen += 1
            if len(self.x_data) < self.capacity:
                self.x_data.append(x_new[i].detach().cpu())
                self.y_data.append(y_new[i].detach().cpu())
            else:
                j = random.randint(0, self.items_seen - 1)
                if j < self.capacity:
                    self.x_data[j] = x_new[i].detach().cpu()
                    self.y_data[j] = y_new[i].detach().cpu()

    def sample_batch(self, batch_size, device):
        if len(self.x_data) == 0:
            return None, None
        
        sample_size = min(batch_size, len(self.x_data))
        indices = random.sample(range(len(self.x_data)), sample_size)
        
        x_batch = torch.stack([self.x_data[i] for i in indices]).to(device)
        y_batch = torch.stack([self.y_data[i] for i in indices]).to(device)
        return x_batch, y_batch
