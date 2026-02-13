#!/bin/bash
cd "$(dirname "$0")"
echo "Recreating venv with system Python (upgrade to 3.10+ recommended)..."
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --quiet --disable-pip-version-check --upgrade
echo "Starting FinanceBot (http://localhost:8501)"
exec streamlit run app.py --server.headless true --server.port 8501
