from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import joblib
import numpy as np


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parents[1]


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


model = joblib.load(BASE_DIR / "model" / "random_forest_model.pkl")

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
    
    input_data = np.array([[ph, Hardness, Solids, Chloramines, Sulfate,
                            Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
    pred = model.predict(input_data)[0]
    result = "✅ Safe to Drink" if pred == 1 else "❌ Not Safe to Drink"

    return templates.TemplateResponse("index.html", {"request": request, "result": result})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
