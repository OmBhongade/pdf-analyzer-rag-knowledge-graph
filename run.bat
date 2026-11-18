@echo off
if not exist .venv\Scripts\python.exe py -3.11 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app\main.py
