from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
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
    if not engine.face_service.recognition_available:
        st.error(f"Face recognition unavailable. {engine.face_service.initialization_error}")
        return
    rgb = np.array(image.convert("RGB"))
    annotated, hits = engine.recognize(rgb)
    st.image(annotated, caption="Recognition result", width="stretch")
    if hits:
        for hit in hits:
            message = hit.attendance_message or "Face matched"
            if hit.status == "IN":
                st.success(
                    f"{hit.employee_name} ({hit.employee_code}) - {hit.department or 'No department'} | confidence: {hit.similarity:.2f} | {message}"
                )
            elif hit.status == "OUT":
                st.warning(
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
    st.dataframe(display_df, hide_index=True, width="stretch")


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
            if not ctx.video_processor.engine.face_service.recognition_available:
                st.error(f"Face recognition unavailable. {ctx.video_processor.engine.face_service.initialization_error}")
            st.metric("Inference FPS", f"{ctx.video_processor.fps:.1f}")
            if ctx.video_processor.last_hits:
                for hit in ctx.video_processor.last_hits:
                    message = hit.attendance_message or "Face matched"
                    line = f"{hit.employee_name} | {hit.employee_code} | {hit.department or 'No department'} | {hit.similarity:.2f} | {message}"
                    if hit.status == "IN":
                        st.success(line)
                    elif hit.status == "OUT":
                        st.warning(line)
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

    st.divider()
    section_title("Edit Attendance By Date")
    selected_date = st.date_input("Select attendance date", value=date.today())
    records_df = attendance_service.attendance_history(start_date=selected_date, end_date=selected_date)
    if records_df.empty:
        st.info("No attendance records found for the selected date.")
        return

    st.dataframe(
        records_df[
            [
                "id",
                "employee_id",
                "name",
                "department",
                "attendance_date",
                "check_in_at",
                "check_out_at",
                "status",
                "confidence",
                "camera_id",
                "device",
                "location",
                "face_sample",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    record_lookup = {
        int(row.id): f'{int(row.employee_id)} - {row.name} | {row.status} | {row.attendance_date}'
        for row in records_df.itertuples(index=False)
    }
    selected_record_id = st.selectbox("Select attendance record", list(record_lookup.keys()), format_func=lambda key: record_lookup[key])
    selected_row = records_df.loc[records_df["id"] == selected_record_id].iloc[0]

    current_check_in = selected_row["check_in_at"]
    current_check_out = selected_row["check_out_at"]
    check_in_base = current_check_in if isinstance(current_check_in, datetime) else datetime.combine(selected_date, time(9, 0))
    check_out_base = current_check_out if isinstance(current_check_out, datetime) else datetime.combine(selected_date, time(17, 0))

    with st.form("edit_attendance_form"):
        attendance_value = selected_row["attendance_date"]
        upd_date = st.date_input(
            "Attendance Date",
            value=pd.to_datetime(attendance_value).date() if pd.notna(attendance_value) else selected_date,
        )
        upd_status = st.selectbox("Status", ["checked_in", "checked_out", "late"], index=["checked_in", "checked_out", "late"].index(selected_row["status"]) if selected_row["status"] in {"checked_in", "checked_out", "late"} else 0)
        upd_check_in_date = st.date_input("Check In Date", value=check_in_base.date())
        upd_check_in_time = st.time_input("Check In Time", value=check_in_base.time())
        upd_has_checkout = st.checkbox("Has Check Out", value=pd.notna(selected_row["check_out_at"]))
        upd_check_out_date = st.date_input("Check Out Date", value=check_out_base.date(), disabled=not upd_has_checkout)
        upd_check_out_time = st.time_input("Check Out Time", value=check_out_base.time(), disabled=not upd_has_checkout)
        upd_confidence = st.slider("Confidence", 0.0, 1.0, value=float(selected_row["confidence"] or 0.0), step=0.01)
        upd_camera_id = st.text_input("Camera ID", value=selected_row["camera_id"] or "")
        upd_device = st.text_input("Device", value=selected_row["device"] or "")
        upd_location = st.text_input("Location", value=selected_row["location"] or "")
        upd_notes = st.text_area("Notes", value="")
        save_attendance = st.form_submit_button("Save Attendance Changes", width="stretch")

    if save_attendance:
        upd_status = "checked_out" if upd_has_checkout else upd_status
        check_in_at = datetime.combine(upd_check_in_date, upd_check_in_time)
        check_out_at = datetime.combine(upd_check_out_date, upd_check_out_time) if upd_has_checkout else None
        ok, msg = attendance_service.update_attendance(
            int(selected_record_id),
            attendance_date=upd_date,
            check_in_at=check_in_at,
            check_out_at=check_out_at,
            status=upd_status,
            confidence=float(upd_confidence),
            camera_id=upd_camera_id.strip() or None,
            device=upd_device.strip() or None,
            location=upd_location.strip() or None,
            notes=upd_notes.strip() or None,
        )
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)
