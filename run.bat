@echo off
echo ===================================================
echo 🫀 ContinualTwin: End-to-End Execution Pipeline
echo ===================================================
echo.
echo [1/4] Parsing PTB-XL Dataset...
python src\process_local_data.py

echo.
echo [2/4] Training Foundation Model and EWC Adaptation...
python main.py

echo.
echo [3/4] Generating Anomaly Detection Graphs...
python demo_anomaly.py

echo.
echo [4/4] Launching Interactive Dashboard...
streamlit run app.py
