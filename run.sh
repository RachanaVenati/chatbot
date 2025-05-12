#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env .env  # Optional: create a dummy .env
streamlit run UIbot.py
