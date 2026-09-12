# 🫀 ContinualTwin: Personalized Cardiac Digital Twin
### Track 5: Continual Learning Adaptation from Population to Individual

## Project Overview
ContinualTwin solves **Catastrophic Forgetting** in healthcare models. We trained a 1D ResNet foundation model on 21,800 patients from the PTB-XL database. When adapting the model to a new patient's longitudinal ECG stream (Digital Twin), Naive Fine-Tuning destroyed the population knowledge (dropping from 87% to 73%). By implementing **Elastic Weight Consolidation (EWC)**, we achieved 100% personalization on the patient while retaining 85% of the foundation model's general knowledge.

## How to Run the Dashboard
1. Install dependencies:
   pip install torch torchvision pandas wfdb streamlit plotly numpy pyyaml
2. Run the interactive web dashboard:
   streamlit run app.py

## Project Structure
- pp.py: The interactive Streamlit dashboard (Live UI)
- main.py: The training loop for Population Pretraining and Continual Learning adaptation
- src/model.py: The lightweight 1D ResNet architecture
- src/ewc.py: The Elastic Weight Consolidation (Fisher Information Matrix) algorithm
- ContinualTwin_Report.md: Our official 4-page academic submission report
