import os
import pandas as pd
import wfdb
import torch
import ast
import numpy as np

def prep_real_data():
    os.makedirs('data', exist_ok=True)
    print("1. Downloading PTB-XL metadata (this is quick)...")
    csv_url = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
    
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        print("Failed to download directly, trying alternate method...")
        import urllib.request
        urllib.request.urlretrieve(csv_url, "data/ptbxl_database.csv")
        df = pd.read_csv("data/ptbxl_database.csv")

    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    print("2. Finding our Population and Patient Stream data...")
    # Find patients with multiple recordings for our Continual Learning stream
    patient_counts = df.patient_id.value_counts()
    multi_record_patients = patient_counts[patient_counts >= 3].index.tolist()
    
    # Pick 5 patients for the stream
    stream_patients = multi_record_patients[:5]
    stream_df = df[df.patient_id.isin(stream_patients)].copy()
    
    # Pick 500 different patients for population pretraining
    pop_df = df[~df.patient_id.isin(stream_patients)].sample(500, random_state=42).copy()
    
    print("3. Streaming ECG waveforms directly from PhysioNet...")
    def fetch_waveforms(dataframe):
        x_data = []
        # We will simplify labels to a binary task (Normal vs Abnormal) for hackathon speed
        # NORM = Normal ECG, everything else is abnormal
        y_data = [] 
        
        for idx, row in dataframe.iterrows():
            # e.g., 'records100/00000/00001_lr'
            file_path = row['filename_lr'] 
            dir_path = '/'.join(file_path.split('/')[:-1])
            record_name = file_path.split('/')[-1]
            
            try:
                # Stream directly from Physionet Servers!
                record = wfdb.rdsamp(record_name, pn_dir=f'ptb-xl/1.0.3/{dir_path}')
                signal = record[0] # Shape: (1000, 12)
                # Convert to (12, 1000) for PyTorch 1D CNN
                signal = np.transpose(signal)
                
                is_normal = 1 if 'NORM' in row['scp_codes'] else 0
                
                x_data.append(torch.tensor(signal, dtype=torch.float32))
                y_data.append(torch.tensor(is_normal, dtype=torch.long))
            except Exception as e:
                print(f"Skipping {record_name}: {e}")
                
        return torch.stack(x_data), torch.stack(y_data)

    print("Fetching Population Data (500 records)... this takes ~1 minute.")
    pop_x, pop_y = fetch_waveforms(pop_df)
    torch.save((pop_x, pop_y), 'data/population.pt')
    
    print("Fetching Patient Stream Data... this takes ~10 seconds.")
    stream_x, stream_y = fetch_waveforms(stream_df)
    # Group stream data by patient
    patient_streams = {}
    for p_id in stream_patients:
        p_mask = (stream_df.patient_id == p_id).values
        # Convert boolean mask to list of indices
        indices = [i for i, x in enumerate(p_mask) if x]
        if len(indices) > 0:
            patient_streams[str(p_id)] = (stream_x[indices], stream_y[indices])
            
    torch.save(patient_streams, 'data/patient_streams.pt')
    print("? Success! Real PTB-XL data is saved in the /data folder.")

if __name__ == '__main__':
    prep_real_data()
