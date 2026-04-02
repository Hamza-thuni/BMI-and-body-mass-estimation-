import cv2
import mediapipe as mp

# Check if the 'solutions' module is now available
try:
    mp_hands = mp.solutions.hands
    print("Mediapipe solutions loaded successfully!")
    print(f"OpenCV version: {cv2.__version__}")
except AttributeError:
    print("Error: 'solutions' module still not found. Try restarting your terminal.")