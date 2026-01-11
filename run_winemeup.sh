#!/bin/bash

echo "🍷 Starting WineMeUp Production Stack..."

# 1. Start Redis (if not running)
# 'service' command works best in WSL with sudo
if ! pgrep redis-server > /dev/null; then
    echo "Starting Redis..."
    sudo service redis-server start
else
    echo "✅ Redis is already running."
fi

# 2. Start FastAPI Backend (Port 8000)
# nohup: Run in background | > api.log: Save logs | &: Don't block terminal
echo "Starting FastAPI Backend..."
nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > logs_api.txt 2>&1 &
API_PID=$!
echo "   -> API started (PID: $API_PID)"

# 3. Start Streamlit Frontend (Port 8501)
echo "Starting Streamlit Frontend..."
nohup streamlit run app/frontend.py --server.port 8501 --server.baseUrlPath /datascience/winemeup/ > logs_frontend.txt 2>&1 &
FRONTEND_PID=$!
echo "   -> Frontend started (PID: $FRONTEND_PID)"

echo "🎉 Deployment Complete! Services are running in the background."
echo "   - API Logs:      tail -f logs_api.txt"
echo "   - Frontend Logs: tail -f logs_frontend.txt"
