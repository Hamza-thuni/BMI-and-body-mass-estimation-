# BMI & Body Composition Analyzer v9 — Demo App

A high-fidelity, laptop-based Flask UI to demonstrate the BMI and Body Composition Analysis system.

## Setup Instructions

### 1. Create a Virtual Environment
It is highly recommended to run this in an isolated virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure the Environment
A `.env` file is included by default. It contains:
```env
APP_MODE=DEV
CAMERA_INDEX=0
MOCK_DELAY=2.0
```
- `CAMERA_INDEX=0` uses your laptop's default built-in webcam.
- `MOCK_DELAY=2.0` simulates the inference delay.

### 4. Run the Application
Start the Flask-SocketIO server:
```bash
python demo_app.py
```

### 5. View the Dashboard
Open your web browser and navigate to:
[http://localhost:5000](http://localhost:5000)

---

## Keyboard Shortcuts
Once the dashboard is open, you can use the following shortcuts for a seamless presentation:
- **`SPACE`** : Initialize Scan (Triggers the inference pipeline)
- **`R`** : Reset UI (Clears the current scan results)

---

## How It Works (The Mock Pipeline)
To allow rapid UI development and testing on any device, this demo uses `MockBMIPipeline` (found in `mock_models.py`). 
This class simulates the V9 hybrid inference pipeline by:
1. "Thinking" for a few seconds.
2. Generating realistic weight, height, and BMI metrics.
3. Calculating **Body Fat Percentage** and Lean Mass using the Deurenberg equation based on the patient's Age and Biological Sex.

**To switch to the real V9 pipeline:**
When deploying to the Raspberry Pi, simply replace the `pipeline.predict()` call in `demo_app.py` with your real ML inference functions.
