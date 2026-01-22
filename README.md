# Water Quality Prediction & Monitoring Dashboard

A small web app that predicts whether water is safe to drink based on common chemical and physical water-quality parameters.

The backend is built with **FastAPI** and serves a simple **HTML (Jinja2)** form. The prediction is made using a pre-trained **scikit-learn** model stored at `model/random_forest_model.pkl`.

## Features

- Web form to enter water-quality measurements
- FastAPI endpoint that runs a model prediction
- Result shown immediately on the same page (Safe / Not Safe)

## Tech Stack

- Python 3.10+ (tested with Python 3.11)
- FastAPI + Uvicorn
- Jinja2 templates
- NumPy, scikit-learn, joblib

## Project Structure

```text
.
├─ app/
│  ├─ main.py
│  └─ utils.py
├─ data/
├─ model/
│  └─ random_forest_model.pkl
├─ notebooks/
├─ static/
│  ├─ script.js
│  └─ styles.css
└─ templates/
	└─ index.html
```

## Setup

### 1) Create and activate a virtual environment (recommended)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

From the project root:

```powershell
python -m pip install -r requirements.txt
```

## Run the App

### Option A (recommended): Run with Uvicorn

From the project root:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

### Option B: Run the module directly

This starts Uvicorn using the `__main__` block.

From the project root:

```powershell
python app\main.py
```

## Model

The app expects a trained model at:

```text
model/random_forest_model.pkl
```

If you don’t have the model yet, train it using the notebook:

- `notebooks/02_model_training.ipynb`


```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## License

Add a license here if you plan to publish/distribute this project.
