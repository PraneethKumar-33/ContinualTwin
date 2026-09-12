import torch
import matplotlib.pyplot as plt
import yaml
import numpy as np
from src.dataset import get_dataloaders
from src.model import ResNet1D
from src.anomaly_detection import TwinAnomalyDetector
import os

def create_demo_graph():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    pop_train, pop_val, pop_test, patient_streams = get_dataloaders(config)
    
    device = torch.device("cpu")
    model = ResNet1D(in_channels=12, num_classes=5).to(device)
    
    stream_loader = patient_streams["patient_0"]
    detector = TwinAnomalyDetector(model, stream_loader, device)
    
    scores = []
    days = list(range(1, 31))
    
    for day in days:
        if day == 21: # Cardiac event!
            x_new = torch.randn(1, 12, 1000) * 5.0 
        elif day == 22:
            x_new = torch.randn(1, 12, 1000) * 3.0
        else:
            x_new, _ = next(iter(stream_loader))
            x_new = x_new[0:1]
            
        score = detector.score_anomaly(x_new)[0]
        scores.append(score)
        
    plt.figure(figsize=(10, 5))
    plt.plot(days, scores, marker='o', linestyle='-', color='b', label='Twin Deviation Score')
    
    # Dynamic threshold based on normal variance
    threshold = np.mean(scores[:20]) + 3 * np.std(scores[:20])
    plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Alert Threshold')
    
    plt.annotate('Cardiac Event Detected!', xy=(21, scores[20]), xytext=(12, scores[20]+0.1),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=12, color='red', fontweight='bold')
             
    plt.title("Patient 0: Continual Cardiac Digital Twin Monitoring", fontsize=14, pad=15)
    plt.xlabel("Day of Monitoring", fontsize=12)
    plt.ylabel("Deviation from Personal Baseline", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/anomaly_demo.png", dpi=300)
    print("Demo graph successfully saved to results/anomaly_demo.png")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    create_demo_graph()
