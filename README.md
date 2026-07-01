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

## Notes

- InsightFace is preferred for 512-d face embeddings.
- If InsightFace is unavailable, the app falls back to an OpenCV-based detector and deterministic embeddings so the UI still runs.
- The SQLite database is created automatically on startup.

## Raspberry Pi Mode

For Raspberry Pi or other low-power devices, set these environment variables in `.env`:

```bash
LOW_POWER_MODE=1
ENABLE_STREAMLIT_WEBRTC=1
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
LIVE_RECOGNITION_INTERVAL_MS=800
ATTENDANCE_REFRESH_SECONDS=4
```

In low-power mode the app:

- skips the heavyweight InsightFace model by default
- uses OpenCV face detection only
- reduces webcam resolution and recognition frequency
- refreshes the marked-attendance table less often to save CPU
# dev-by Mohammed Kahab K 

