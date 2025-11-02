from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

# Initialize
app = FastAPI()

# Setup static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
model = joblib.load("model/random_forest_model.pkl")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    ph: float = Form(...),
    Hardness: float = Form(...),
    Solids: float = Form(...),
    Chloramines: float = Form(...),
    Sulfate: float = Form(...),
    Conductivity: float = Form(...),
    Organic_carbon: float = Form(...),
    Trihalomethanes: float = Form(...),
    Turbidity: float = Form(...)
):
    # Make prediction
    input_data = np.array([[ph, Hardness, Solids, Chloramines, Sulfate,
                            Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
    pred = model.predict(input_data)[0]
    result = "✅ Safe to Drink" if pred == 1 else "❌ Not Safe to Drink"

    return templates.TemplateResponse("index.html", {"request": request, "result": result})
