import torch
import copy

class EWC:
    def __init__(self, model, dataloaders, device):
        self.model = model
        self.device = device
        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]
        self.dataloaders = dataloaders

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        
        # Save a copy of current weights to protect
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._means[n] = p.data.clone()
                
        self._precision_matrices = self._diag_fisher()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p.data)

        self.model.eval()
        n_batches = sum(len(dl) for dl in self.dataloaders)
        if n_batches == 0:
            return precision_matrices

        for dataloader in self.dataloaders:
            for x, y in dataloader:
                self.model.zero_grad()
                x = x.to(self.device)
                output, _ = self.model(x)

                log_probs = torch.nn.functional.log_softmax(output, dim=1)
                probs = log_probs.exp().detach()
                samples = torch.multinomial(probs, 1).squeeze(1)
                loss = torch.nn.functional.nll_loss(log_probs, samples)
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        precision_matrices[n] += p.grad.data ** 2 / n_batches

        return precision_matrices

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self._precision_matrices and n in self._means:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
