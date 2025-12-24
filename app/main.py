from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.api.predict import router as predict_router
from app.core.config import OUTPUT_DIR

app = FastAPI(
    title="Indian Flower Recognition API",
    description="An API to recognize Indian flowers using a pre-trained MobileNet model.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# Serve gradcam images HERE
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# API routes
app.include_router(predict_router, tags=["Prediction"])

@app.get("/")
def root():
    return {"status": "OK"}

@app.get("/health")
def health():
    return {"status": "ok"}
