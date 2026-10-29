# Attendance Management System

Production-oriented facial-recognition attendance system built with Python, Streamlit, OpenCV, InsightFace, SQLAlchemy, and SQLite.

## Features

- Secure JWT-based authentication with bcrypt password hashing
- Admin and employee roles
- Face registration with quality checks
- Live recognition and automatic attendance marking
- Attendance analytics and export to CSV, Excel, and PDF
- Employee, user, camera, and settings management
- SQLite-backed persistence with SQLAlchemy ORM

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optionally create a `.env` file from the example below.
4. Run the app:

```bash
streamlit run app.py
```

## Default Admin

If no users exist, the app seeds a default admin:

- Username: `admin`
- Password: `Admin@1234!`

Change these credentials immediately after first login.

## Face recognition and Raspberry Pi setup

The default CPU backend uses OpenCV YuNet detection, landmark alignment, and
SFace learned embeddings. It runs without InsightFace or ONNX Runtime.

```bash
python -m pip install -r requirements-pi.txt
python scripts/download_face_models.py
streamlit run app.py
```

The setup script downloads the official OpenCV Zoo model binaries and their
licenses, verifies each binary against its Git LFS SHA-256 and size, and installs
models atomically in `assets/models/`. Model files are excluded from Git; run the
setup script on each deployment. Restart the app after installing models.

In `.env`, you can set:

```bash
LOW_POWER_MODE=1
FACE_BACKEND=sface
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
MAX_FRAME_WIDTH=640
LIVE_RECOGNITION_INTERVAL_MS=250
RECOGNITION_CONFIRMATIONS=3
RECOGNITION_MARGIN=0.08
FACE_DETECTION_CONFIDENCE=0.9
ATTENDANCE_REFRESH_SECONDS=4
```

Low-power mode caps OpenCV at one CPU thread. Recognition runs at a bounded
interval while intermediate frames pass through as a live preview. Busy inference
does not queue extra work. The displayed inference FPS measures model processing,
not camera frame rate. Actual recognition frequency also includes the configured
interval. Raise the interval if the Pi overheats or inference consumes too much CPU.

Matching caches normalized enrollment vectors for 10 seconds and uses matrix
multiplication. A match must exceed the configured recognition threshold and beat
the next *different employee* by the configured margin. Live attendance requires
three consecutive matching inference frames by default. Unknown or missed frames
reset confirmation. Snapshot recognition uses one frame. This is identity
confirmation, not liveness or anti-spoofing detection.

### Add faces

1. Create an active employee in **Employees**.
2. Open **Register Face**, select the employee, and upload or capture photos.
3. Add 3-5 sharp photos: straight ahead and slightly left/right in normal lighting.
4. Click **Register Faces**. Rejected photos show the reason; accepted samples
   become available to live recognition within 10 seconds.

Only one face is accepted per enrollment photo. Quality checks examine the face
crop for size, brightness, and blur. Inconsistent faces in one batch are rejected.
Blur is measured at a fixed 112x112 face size so photo resolution does not
change the check. Faces need the configured minimum pixel size, but do not have
to fill a fixed percentage of the whole photo. Default quality thresholds are
`FACE_BLUR_THRESHOLD=35` and `FACE_MIN_BRIGHTNESS=40`; restart the app after
changing these environment variables. Rejection messages identify the photo,
explain how to retake it, and offer a preview. Batch consistency compares the
closest accepted photo rather than their average, accommodating varied angles
without changing the identity recognition threshold.
Enrollment reuses detections and caches its model per browser session.

Existing enrollment vectors are tagged `legacy` and excluded from matching;
**re-enroll all employees after this upgrade**. Models have separate tags so
incompatible embedding spaces cannot mix. Old records and attendance history
remain intact. NumPy image inputs to `FaceService` must be RGB.

`FACE_BACKEND=insightface` remains available with the full `requirements.txt`
installation, using CPU detection and recognition modules only. Switching backends
requires enrollment for that backend. Missing models disable recognition and
enrollment with a setup message; the app never falls back to pixel-based identity
matching.

The recognition threshold can be adjusted in Settings. Defaults are starting
points, not measured accuracy guarantees. Validate false matches, missed matches,
latency, and memory with your camera, lighting, and expected employee count on
the target Raspberry Pi before deployment.

Model APIs and examples: [OpenCV face recognition tutorial](https://docs.opencv.org/4.6.0/d0/dd4/tutorial_dnn_face.html),
[OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface),
[OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

### Validation

```bash
python -m unittest discover -s tests -v
```

Model smoke tests run when the downloaded model files are present. Tests use
in-memory databases, not the attendance database.

# dev-by Mohammed Kahab K
