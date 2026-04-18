from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes import router
import os

app = FastAPI(title="Insurance Fraud Detection API")

# Allow Streamlit to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve outputs folder as static files so Streamlit can load images
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

app.include_router(router)


@app.get("/")
def health_check():
    return {"status": "running"}