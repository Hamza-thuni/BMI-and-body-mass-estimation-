import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # DEV = Laptop Webcam, PROD = Raspberry Pi CSI
    APP_MODE = os.getenv("APP_MODE", "DEV").upper()
    
    # Try parsing CAMERA_INDEX as integer, if it fails, it's likely a string path (e.g. picamera2)
    try:
        CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
    except ValueError:
        CAMERA_INDEX = os.getenv("CAMERA_INDEX", "0")
        
    MOCK_DELAY = float(os.getenv("MOCK_DELAY", "2.0"))
    
    # Height constraints for manual input fallback (cm)
    MIN_HEIGHT = 100
    MAX_HEIGHT = 250
