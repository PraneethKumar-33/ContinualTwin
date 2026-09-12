import os
import pandas as pd
import wfdb
import torch
import ast
import numpy as np
from tqdm import tqdm

def prep_local_data():
    base_dir = r'C:\Users\HP\Downloads\DL Hack Dataset'
    
    found_csv = False
    for root, dirs, files in os.walk(base_dir):
        if 'ptbxl_database.csv' in files:
            base_dir = root
            found_csv = True
            break
            
    if not found_csv:
        print("ERROR: Could not find ptbxl_database.csv in DL Hack Dataset!")
        return
                
    df = pd.read_csv(os.path.join(base_dir, 'ptbxl_database.csv'))
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    patient_counts = df.patient_id.value_counts()
    multi_record_patients = patient_counts[patient_counts >= 3].index.tolist()
    
    stream_patients = multi_record_patients[:5]
    stream_df = df[df.patient_id.isin(stream_patients)].copy()
    
    # WE ARE USING ALL REMAINING PATIENTS! (~21,800 records)
    pop_df = df[~df.patient_id.isin(stream_patients)].copy()
    
    def load_waveforms(dataframe, desc):
        x_data, y_data = [], []
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=desc, unit="ECGs"):
            record_path = os.path.join(base_dir, row['filename_lr'])
            try:
                record = wfdb.rdsamp(record_path)
                signal = np.transpose(record[0])
                is_normal = 1 if 'NORM' in row['scp_codes'] else 0
                
                x_data.append(torch.tensor(signal, dtype=torch.float32))
                y_data.append(torch.tensor(is_normal, dtype=torch.long))
            except Exception as e:
                pass
        return torch.stack(x_data), torch.stack(y_data)

    print(f"\n[Phase 1/2] Processing FULL Population Data ({len(pop_df)} Records)...")
    pop_x, pop_y = load_waveforms(pop_df, "Population Data")
    torch.save((pop_x, pop_y), 'data/population.pt')
    
    print("\n[Phase 2/2] Processing Patient Stream Data locally...")
    stream_x, stream_y = load_waveforms(stream_df, "Patient Streams")
    
    patient_streams = {}
    for p_id in stream_patients:
        p_mask = (stream_df.patient_id == p_id).values
        indices = [i for i, x in enumerate(p_mask) if x]
        if len(indices) > 0:
            patient_streams[str(p_id)] = (stream_x[indices], stream_y[indices])
            
    torch.save(patient_streams, 'data/patient_streams.pt')
    print("\n✅ Success! Massive Local Dataset parsed and saved.")

if __name__ == '__main__':
    prep_local_data()
