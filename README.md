# TrendScope

This project consists of a Python backend that runs using Flask/FastAPI. Follow to the steps below to run it locally.

### 1. Prerequisites
- Python 3.9+ installed on your machine.

### 2. Set up the Backend
First, navigate to the `backend` directory in your terminal:
```bash
cd backend
```

Create and activate a Python virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Configuration
Ensure you have a `.env` file present in the `backend` directory containing the necessary environment variables (this file gets loaded automatically by `run.py`).

### 5. Run the Server
Start the application using the `run.py` script:
```bash
python run.py
```
