#!/bin/bash

source ~/VirtualEnv/fyp_ai_env/bin/activate
cd ~/Year_2024/apps/streamlit_apps

# streamlit run app.py --server.port 2701 >> ./logs/streamlit.log &
# Attempt to run to log file with process running in background	
nohup streamlit run app.py --server.port 2701 >> ./logs/streamlit.log 2>&1 &
