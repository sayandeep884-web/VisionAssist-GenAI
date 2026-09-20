from dotenv import load_dotenv
import os

load_dotenv()      # loads the values from .env file.

APP_NAME = os.getenv("APP_NAME", "VisionAssist GenAI")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")