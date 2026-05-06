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
    """Video streaming generator function with V9-style overlays."""
    import time
    import numpy as np
    import mediapipe as mp
    import live_bmi_demo_v9 as v9
    
    cam = get_camera()
    if not cam.isOpened():
        print("[Video Feed] Error: Camera could not be opened.")
    
    while True:
        success, frame = cam.read()
        if not success:
            # ... (error handling remains same)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (0, 0, 150)
            cv2.putText(frame, "CAMERA ERROR", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1.0)
            continue
            
        # Save raw frame for inference
        stream_state['latest_frame'] = frame.copy()
        display = frame.copy()
        h_f, w_f = frame.shape[:2]

        # 1. ArUco detection & drawing
        corners, ids = v9.detect_aruco(frame)
        if ids is not None:
            ids_f = ids.flatten()
            if 0 in ids_f and 1 in ids_f:
                idx0 = np.where(ids_f == 0)[0][0]
                idx1 = np.where(ids_f == 1)[0][0]
                c0 = corners[idx0][0].mean(axis=0)
                c1 = corners[idx1][0].mean(axis=0)
                
                # Draw pink scale line
                cv2.line(display, (int(c0[0]), int(c0[1])), (int(c1[0]), int(c1[1])), (255, 0, 255), 2)
                cv2.putText(display, "ID0 (80cm)", (int(c0[0]) + 10, int(c0[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.putText(display, "ID1 (180cm)", (int(c1[0]) + 10, int(c1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                pixel_sep = abs(c1[1] - c0[1])
                px_per_m = pixel_sep / 1.0 # MARKER_SEPARATION_M
                stream_state['global_scale'] = px_per_m
                cv2.putText(display, f"Scale: {px_per_m:.1f} px/m", (10, h_f - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 2. Pose detection & drawing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            res = pipeline.landmarker.detect(mp_img)
            if res and res.pose_landmarks:
                for lm in res.pose_landmarks[0]:
                    cx, cy = int(lm.x * w_f), int(lm.y * h_f)
                    cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)
        except:
            pass

        ret, buffer = cv2.imencode('.jpg', display)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.01)

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
