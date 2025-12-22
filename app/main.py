from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.predict import router as predict_router
from app.core.config import OUTPUT_DIR

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title = "Indian Flower Recognition API",
    description = "An API to recognize Indian flowers using a pre-trained MobileNet model.",
    version = "1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Serve frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


#serve Grad-CAM images
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# Register API Routers  
app.include_router(predict_router, tags=["Prediction"])

@app.get("/")
def root():
    return {
        "status": "OK",
        "message" : "Indian Flower Recognition API is running."
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}
