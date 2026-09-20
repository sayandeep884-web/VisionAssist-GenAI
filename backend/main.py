from fastapi import FastAPI
from backend.config import APP_NAME, ENVIRONMENT

app = FastAPI(title=APP_NAME)


@app.get("/")
def root():
    return {
        "message": f"{APP_NAME} API is running!",
        "environment": ENVIRONMENT
    }