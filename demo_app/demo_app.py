from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
from config import Config
from mock_models import MockBMIPipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_bmi_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the mock pipeline
pipeline = MockBMIPipeline()

camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(Config.CAMERA_INDEX)
    return camera

def gen_frames():
    """Video streaming generator function."""
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            # Optionally encode directly, or do light processing if needed
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('trigger_scan')
def handle_trigger_scan(data):
    """
    Called when the frontend requests a scan.
    Runs the mock pipeline in a background thread to avoid blocking the server.
    """
    age = data.get('age', 25)
    sex_is_male = data.get('sex', 'male') == 'male'
    
    print(f"[SocketIO] Scan triggered for Age: {age}, Sex: {'Male' if sex_is_male else 'Female'}")
    
    # Notify frontend that scan has started (e.g. to show 'Scanning...' animation)
    socketio.emit('scan_started')
    
    # Background task to run inference
    def run_inference():
        result = pipeline.predict(age=age, sex_is_male=sex_is_male)
        socketio.emit('scan_result', result)
        print(f"[SocketIO] Scan complete: {result}")
        
    socketio.start_background_task(run_inference)

@socketio.on('reset')
def handle_reset():
    print("[SocketIO] UI Reset requested")
    socketio.emit('ui_reset')

if __name__ == '__main__':
    print(f"=== Starting BMI Demo Server ({Config.APP_MODE} mode) ===")
    print(f"Camera Index: {Config.CAMERA_INDEX}")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
