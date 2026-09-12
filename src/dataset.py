import torch
from torch.utils.data import TensorDataset, DataLoader

def get_dataloaders(config):
    pop_x, pop_y = torch.load('data/population.pt')
    patient_streams_data = torch.load('data/patient_streams.pt')
    
    # 70% Train, 15% Val, 15% Test
    total = len(pop_x)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)
    
    pop_train_x, pop_train_y = pop_x[:train_end], pop_y[:train_end]
    pop_val_x, pop_val_y = pop_x[train_end:val_end], pop_y[train_end:val_end]
    pop_test_x, pop_test_y = pop_x[val_end:], pop_y[val_end:]
    
    b_size = config['training']['batch_size']
    pop_train_loader = DataLoader(TensorDataset(pop_train_x, pop_train_y), batch_size=b_size, shuffle=True)
    pop_val_loader = DataLoader(TensorDataset(pop_val_x, pop_val_y), batch_size=b_size, shuffle=False)
    pop_test_loader = DataLoader(TensorDataset(pop_test_x, pop_test_y), batch_size=b_size, shuffle=False)
    
    patient_streams = {}
    first_patient_id = list(patient_streams_data.keys())[0]
    stream_x, stream_y = patient_streams_data[first_patient_id]
    
    patient_streams["patient_0"] = DataLoader(TensorDataset(stream_x, stream_y), batch_size=2, shuffle=False)
    
    return pop_train_loader, pop_val_loader, pop_test_loader, patient_streams
