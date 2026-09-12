import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import copy
from src.dataset import get_dataloaders
from src.model import ResNet1D
from src.replay_buffer import ReplayBuffer
from src.ewc import EWC

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs, _ = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    pop_train_loader, pop_val_loader, pop_test_loader, patient_streams = get_dataloaders(config)
    
    from tqdm import tqdm
    
    print("\n=== PHASE 1: Population Pretraining ===")
    base_model = ResNet1D(in_channels=config['dataset']['channels'], num_classes=config['dataset']['classes']).to(device)
    optimizer = optim.Adam(base_model.parameters(), lr=config['training']['lr'])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config['training']['population_epochs']):
        base_model.train()
        # Added beautiful progress bar!
        pbar = tqdm(pop_train_loader, desc=f"Epoch {epoch+1}/{config['training']['population_epochs']} [Phase 1]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs, _ = base_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        val_acc = evaluate(base_model, pop_val_loader, device)
        print(f"✅ Epoch {epoch+1} - Validation Accuracy: {val_acc:.2f}%\n")
    
    pop_acc = evaluate(base_model, pop_test_loader, device)
    print(f"Population Base Model Test Accuracy: {pop_acc:.2f}%")
    
    patient_id = "patient_0"
    stream_loader = patient_streams[patient_id]
    
    methods = ["Naive", "Replay", "EWC"]
    results = {}
    
    for method in methods:
        print(f"\n=== PHASE 2: Personalization -> Method: {method} ===")
        model = copy.deepcopy(base_model)
        optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
        
        buffer = ReplayBuffer(config['cl_methods']['replay_buffer_size'])
        if method == "Replay":
            for x, y in pop_train_loader:
                buffer.add_data(x, y)
                if buffer.items_seen > buffer.capacity: break
                
        ewc = EWC(model, pop_train_loader, device) if method == "EWC" else None
        
        for step, (x, y) in enumerate(stream_loader):
            x, y = x.to(device), y.to(device)
            
            pbar = tqdm(range(config['training']['stream_epochs_per_step']), desc=f"Adapting to Stream Batch {step+1}", leave=False)
            for epoch in pbar:
                model.train()
                optimizer.zero_grad()
                
                outputs, _ = model(x)
                loss = criterion(outputs, y)
                
                if method == "Replay":
                    buf_x, buf_y = buffer.sample_batch(len(x), device)
                    if buf_x is not None:
                        buf_out, _ = model(buf_x)
                        loss += criterion(buf_out, buf_y)
                        
                if method == "EWC":
                    loss += config['cl_methods']['ewc_lambda'] * ewc.penalty(model)
                    
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        patient_acc = evaluate(model, stream_loader, device)
        pop_retention_acc = evaluate(model, pop_test_loader, device)
        
        print(f"[{method}] Patient Acc: {patient_acc:.2f}% | Population Retention (Forgetting): {pop_retention_acc:.2f}%")
        results[method] = {'patient': patient_acc, 'population': pop_retention_acc}
        
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Method':<10} | {'Patient Acc (Personalization)':<30} | {'Population Acc (Retention)':<30}")
    print("-" * 75)
    for method, res in results.items():
        print(f"{method:<10} | {res['patient']:<30.2f} | {res['population']:<30.2f}")

if __name__ == '__main__':
    main()
