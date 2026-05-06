from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
from config import Config
from v9_pipeline_wrapper import RealV9Pipeline
import webbrowser
from threading import Timer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_bmi_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for video stream to share with inference thread
stream_state = {
    'latest_frame': None,
    'global_scale': None
}

# Initialize the real V9 pipeline
pipeline = RealV9Pipeline()

camera = None

def get_camera():
    global camera
    if camera is None:
        import platform
        if platform.system() == 'Windows':
            camera = cv2.VideoCapture(Config.CAMERA_INDEX, cv2.CAP_DSHOW)
        else:
            camera = cv2.VideoCapture(Config.CAMERA_INDEX)
    return camera

def gen_frames():
    """Video streaming generator function."""
    import time
    import numpy as np
    cam = get_camera()
    
    if not cam.isOpened():
        print("[Video Feed] Error: Camera could not be opened.")
    
    while True:
        success, frame = cam.read()
        if not success:
            print("[Video Feed] Error: Failed to read frame from camera.")
            # Create a blank red frame with error text
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (0, 0, 150)
            cv2.putText(frame, "CAMERA ERROR", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1.0)
            continue
            
        else:
            # Save frame for inference
            stream_state['latest_frame'] = frame.copy()
            
            # Continuously update the ArUco scale
            scale_data = pipeline.calculate_scale(frame)
            if scale_data is not None:
                stream_state['global_scale'] = scale_data
                
            # Draw overlay
            if stream_state['global_scale'] is not None:
                focal_length, z_depth = stream_state['global_scale']
                cv2.putText(frame, f"Wall Distance: {z_depth:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Waiting for ArUco...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)

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
    Runs the actual V9 pipeline in a background thread to avoid blocking the server.
    """
    age = data.get('age', 25)
    sex_is_male = data.get('sex', 'male') == 'male'
    offset_cm = data.get('offset_cm', 50)
    
    print(f"[SocketIO] Scan triggered for Age: {age}, Sex: {'Male' if sex_is_male else 'Female'}, Offset: {offset_cm}cm")
    
    # Notify frontend that scan has started (e.g. to show 'Scanning...' animation)
    socketio.emit('scan_started')
    
    # Background task to run inference
    def run_inference():
        frame = stream_state['latest_frame']
        
        if stream_state['global_scale'] is None:
            socketio.emit('scan_error', {'message': 'ArUco scale not found. Ensure markers are visible.'})
            return
            
        focal_length, z_depth = stream_state['global_scale']
        
        try:
            result = pipeline.predict(frame=frame, age=age, sex_is_male=sex_is_male, z_depth=z_depth, focal_length=focal_length, offset_cm=offset_cm)
            socketio.emit('scan_result', result)
            print(f"[SocketIO] Scan complete: {result}")
        except Exception as e:
            print(f"[SocketIO] Inference error: {e}")
            socketio.emit('scan_error', {'message': str(e)})
        
    socketio.start_background_task(run_inference)

@socketio.on('reset')
def handle_reset():
    print("[SocketIO] UI Reset requested")
    socketio.emit('ui_reset')

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    print(f"=== Starting BMI Demo Server ({Config.APP_MODE} mode) ===")
    print(f"Camera Index: {Config.CAMERA_INDEX}")
    Timer(1, open_browser).start()
    socketio.run(app, debug=False, use_reloader=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
