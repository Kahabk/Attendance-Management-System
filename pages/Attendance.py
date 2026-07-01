from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from config import settings
from services.attendance_service import attendance_service
from services.camera_service import LiveVideoProcessor, RecognitionEngine
from utils.ui import section_title

try:  # pragma: no cover - optional realtime dependency
    from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
except Exception:  # pragma: no cover
    webrtc_streamer = None
    WebRtcMode = None
    RTCConfiguration = None


def _snapshot_recognition(image: Image.Image) -> None:
    engine = RecognitionEngine()
    rgb = np.array(image.convert("RGB"))
    annotated, hits = engine.recognize(rgb)
    st.image(annotated, caption="Recognition result", use_container_width=True)
    if hits:
        for hit in hits:
            message = hit.attendance_message or "Face matched"
            if hit.status == "marked":
                st.success(
                    f"{hit.employee_name} ({hit.employee_code}) - {hit.department or 'No department'} | confidence: {hit.similarity:.2f} | {message}"
                )
            else:
                st.info(
                    f"{hit.employee_name} ({hit.employee_code}) - {hit.department or 'No department'} | confidence: {hit.similarity:.2f} | {message}"
                )
    else:
        st.info("No known face detected.")


@st.fragment(run_every=settings.attendance_refresh_seconds)
def _render_marked_today_table() -> None:
    section_title("Attendance Marked Today")
    today_df = attendance_service.todays_attendance()
    if today_df.empty:
        st.info("No attendance has been marked today yet.")
        return

    display_df = today_df[
        [
            "employee_id",
            "name",
            "department",
            "mark_state",
            "in_date",
            "in_time",
            "out_date",
            "out_time",
            "confidence",
            "face_embedding_id",
            "face_sample",
            "camera_id",
            "device",
        ]
    ].copy()
    display_df.columns = [
        "Employee Code",
        "Employee Name",
        "Department",
        "Mark",
        "In Date",
        "In Time",
        "Out Date",
        "Out Time",
        "Confidence",
        "Face ID",
        "Face Sample",
        "Camera",
        "Device",
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render(current_user: dict[str, object]) -> None:
    section_title("Live Attendance Recognition")
    st.caption("Face recognition updates attendance automatically. The marked table refreshes on a low-power interval.")

    if settings.enable_streamlit_webrtc and webrtc_streamer is not None:
        ctx = webrtc_streamer(
            key="attendance-live-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": settings.camera_width},
                    "height": {"ideal": settings.camera_height},
                    "frameRate": {"ideal": 12 if settings.low_power_mode else 24},
                },
                "audio": False,
            },
            video_processor_factory=LiveVideoProcessor,
            async_processing=True,
        )
        if ctx.video_processor:
            st.metric("Recognition FPS", f"{ctx.video_processor.fps:.1f}")
            if ctx.video_processor.last_hits:
                for hit in ctx.video_processor.last_hits:
                    message = hit.attendance_message or "Face matched"
                    line = f"{hit.employee_name} | {hit.employee_code} | {hit.department or 'No department'} | {hit.similarity:.2f} | {message}"
                    if hit.status == "marked":
                        st.success(line)
                    else:
                        st.info(line)
            else:
                st.info("No face confirmed yet. When a registered face is recognized, the marked attendance table below updates.")
    else:
        st.warning("Real-time webcam mode is unavailable. Use the snapshot fallback below.")
        snapshot = st.camera_input("Capture a frame")
        if snapshot is not None and st.button("Run Recognition"):
            _snapshot_recognition(Image.open(snapshot))

    _render_marked_today_table()
