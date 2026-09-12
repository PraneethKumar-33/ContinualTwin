# 🫀 ContinualTwin: Personalized Cardiac Digital Twin
### Track 5: Continual Learning Adaptation from Population to Individual

## Project Overview
ContinualTwin solves **Catastrophic Forgetting** in healthcare models. We trained a 1D ResNet foundation model on 21,800 patients from the PTB-XL database. When adapting the model to a new patient's longitudinal ECG stream (Digital Twin), Naive Fine-Tuning destroyed the population knowledge (dropping from 87% to 73%). By implementing **Elastic Weight Consolidation (EWC)**, we achieved 100% personalization on the patient while retaining 85% of the foundation model's general knowledge.

## How to Run (One-Command End-to-End)
This repository satisfies the reproducibility rule (`seed=42` locked). To run the entire pipeline end-to-end (data parsing, model training, graph generation, and dashboard launching), simply execute:

```bash
run.bat
```

*(Note: Ensure `torch torchvision pandas wfdb streamlit plotly numpy pyyaml` are installed first).*

## Project Structure
-  pp.py: The interactive Streamlit dashboard (Live UI)
- main.py: The training loop for Population Pretraining and Continual Learning adaptation
- src/model.py: The lightweight 1D ResNet architecture
- src/ewc.py: The Elastic Weight Consolidation (Fisher Information Matrix) algorithm
- ContinualTwin_Report.md: Our official 4-page academic submission report
