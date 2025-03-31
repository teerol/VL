rd /s /q venv
py -3.11 -m venv ./venv
call venv\Scripts\activate.bat

python --version

pip install -r requirements.txt