# 🫀 ContinualTwin: Personalized Cardiac Digital Twin

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### 🏆 Track 5: Continual Learning Adaptation from Population to Individual

ContinualTwin is a healthcare AI platform that solves **Catastrophic Forgetting** when personalizing wearable health monitors. 

## 🚨 The Problem
When a global Foundation Model (trained on thousands of patients) is fine-tuned to monitor a *single* patient's smartwatch ECG, it suffers from Catastrophic Forgetting. It perfectly memorizes the new patient, but entirely forgets what a general heart attack looks like in other demographics. 

## 💡 Our Solution
We implemented **Elastic Weight Consolidation (EWC)** to create a "Digital Twin". 
By calculating a Fisher Information Matrix during population training, our Continual Learning engine locks the critical neural weights. This allows the model to adapt perfectly to the individual patient *without* destroying its foundational medical knowledge.

---

## 📊 Performance & Results
We trained a 1D ResNet on **21,800 patients** from the PTB-XL clinical database, and then continually adapted it to Patient 0. 

| Adaptation Strategy | Patient Accuracy (Personalization) | Population Retention (Forgetting) |
| :--- | :--- | :--- |
| **Base Model** (No adaptation) | N/A | **87.09%** |
| **Naive Fine-Tuning** | 100.0% | 73.88% 🔻 *(Severe Forgetting)* |
| **Experience Replay** | 100.0% | 86.72% ➖ |
| **EWC (ContinualTwin)** | **100.0%** | **84.89%** 🟢 *(Preserved!)* |

Our EWC algorithm successfully retained ~85% of the foundation model's general knowledge while achieving 100% personalization for the patient's Digital Twin.

## 📉 Real-Time Anomaly Detection
Once the Digital Twin is created, it monitors the patient's incoming ECG stream. Below is our system successfully detecting a massive cardiac deviation on Day 21:

![Digital Twin Anomaly Detection](assets/anomaly_demo.png)

---

## 🚀 How to Run (One-Command End-to-End)
This repository satisfies the hackathon reproducibility rules (Seed=42 locked). To run the entire pipeline end-to-end (data parsing, model training, graph generation, and dashboard launching), simply execute the batch script:

`ash
run.bat
`

*(Note: Ensure 	orch torchvision pandas wfdb streamlit plotly numpy pyyaml are installed first).*

## 📁 Project Structure
- pp.py: The interactive Streamlit dashboard (Live UI)
- main.py: The training loop for Population Pretraining and Continual Learning adaptation
- src/model.py: The lightweight 1D ResNet architecture
- src/ewc.py: The Elastic Weight Consolidation (Fisher Information Matrix) algorithm
- Report.md: Our official academic submission report
