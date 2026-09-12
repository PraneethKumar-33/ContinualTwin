import torch
import torch.nn.functional as F
import numpy as np

class TwinAnomalyDetector:
    def __init__(self, model, baseline_dataloader, device):
        self.model = model
        self.device = device
        self.baseline_embedding = self._compute_baseline(baseline_dataloader)

    def _compute_baseline(self, dataloader):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                _, features = self.model(x)
                embeddings.append(features)
        
        all_embeddings = torch.cat(embeddings, dim=0)
        # Average the embeddings to get the "Digital Twin" baseline center
        baseline_center = torch.mean(all_embeddings, dim=0)
        return baseline_center

    def score_anomaly(self, x_new):
        self.model.eval()
        with torch.no_grad():
            x_new = x_new.to(self.device)
            _, features = self.model(x_new)
            
            # Cosine distance: 1 - cosine_similarity
            similarity = F.cosine_similarity(features, self.baseline_embedding.unsqueeze(0))
            distance = 1.0 - similarity
            return distance.cpu().numpy()
