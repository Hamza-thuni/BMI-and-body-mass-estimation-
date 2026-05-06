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
    'global_scale': None,
    'is_scanning': False,
    'scan_params': {
        'age': 25,
        'sex_is_male': True,
        'parallax_factor': 1.0
    }
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
                px_per_m = stream_state['global_scale']
                cv2.putText(frame, f"Scale: {px_per_m:.1f} px/m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Waiting for ArUco (ID 0 & 1)...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
    Called when the frontend requests to start or update scanning.
    """
    age = data.get('age', 25)
    sex_is_male = data.get('sex', 'male') == 'male'
    parallax_factor = data.get('parallax_factor', 1.0)
    
    stream_state['scan_params'] = {
        'age': age,
        'sex_is_male': sex_is_male,
        'parallax_factor': parallax_factor
    }
    
    if not stream_state['is_scanning']:
        stream_state['is_scanning'] = True
        print(f"[SocketIO] Continuous scanning STARTED for Age: {age}, Sex: {'Male' if sex_is_male else 'Female'}")
        socketio.emit('scan_started')
    else:
        print(f"[SocketIO] Scan parameters UPDATED: Age={age}, Parallax={parallax_factor}")

def background_inference():
    """Background task that runs inference whenever is_scanning is True."""
    import time
    while True:
        if stream_state['is_scanning']:
            frame = stream_state['latest_frame']
            scale = stream_state['global_scale']
            params = stream_state['scan_params']
            
            if frame is not None and scale is not None:
                try:
                    result = pipeline.predict(
                        frame=frame, 
                        age=params['age'], 
                        sex_is_male=params['sex_is_male'], 
                        px_per_m=scale, 
                        parallax_factor=params['parallax_factor']
                    )
                    socketio.emit('scan_result', result)
                except Exception as e:
                    # If it's a real error (like no person), we can notify, 
                    # but for continuous we might want to just skip or send a status
                    # socketio.emit('scan_status', {'message': str(e)})
                    pass
        socketio.sleep(0.2) # Run at ~5 FPS

@socketio.on('stop_scan')
def handle_stop_scan():
    stream_state['is_scanning'] = False
    print("[SocketIO] Continuous scanning STOPPED")
    socketio.emit('scan_stopped')

@socketio.on('reset')
def handle_reset():
    stream_state['is_scanning'] = False
    print("[SocketIO] UI Reset requested")
    socketio.emit('ui_reset')

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    print(f"=== Starting BMI Demo Server ({Config.APP_MODE} mode) ===")
    print(f"Camera Index: {Config.CAMERA_INDEX}")
    Timer(1, open_browser).start()
    
    # Start the continuous inference background task
    socketio.start_background_task(background_inference)
    
    socketio.run(app, debug=False, use_reloader=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
