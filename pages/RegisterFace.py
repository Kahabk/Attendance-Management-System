from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from config import settings
from database import get_db_session
from models import Employee, FaceEmbedding
from services.face_service import FaceService
from utils.logger import logger
from utils.serialization import array_to_blob
from utils.ui import section_title


def _save_image(employee_code: str, image: Image.Image, prefix: str) -> Path:
    folder = settings.upload_dir / "face_registration" / employee_code
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}.png"
    image.save(path)
    return path


def render(current_user: dict[str, object]) -> None:
    if current_user.get("role") != "Admin":
        st.warning("Only administrators can register faces.")
        return

    section_title("Face Registration")
    if "registration_face_service" not in st.session_state:
        st.session_state.registration_face_service = FaceService()
    face_service = st.session_state.registration_face_service
    if not face_service.recognition_available:
        st.error(f"Face recognition is unavailable. {face_service.initialization_error}")
        return
    st.caption("Add 3-5 clear photos: straight ahead and slightly left/right, in normal lighting. Keep only one face in each photo.")

    with get_db_session() as session:
        employees = session.query(Employee).filter(Employee.is_active.is_(True)).order_by(Employee.name.asc()).all()
    if not employees:
        st.info("Create an employee before registering faces.")
        return

    employee_map = {f"{emp.employee_id} - {emp.name}": emp.id for emp in employees}
    selected_label = st.selectbox("Select Employee", list(employee_map.keys()))
    employee_id = employee_map[selected_label]

    uploaded_files = st.file_uploader(
        "Upload face images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload front-facing, well-lit images. Use several clear views for more reliable matching.",
    )
    camera_photo = st.camera_input("Capture a face photo")

    total_samples = len(uploaded_files or []) + (1 if camera_photo else 0)
    st.write(f"Samples selected: {total_samples}")

    if st.button("Register Faces", width="stretch"):
        if total_samples < 1:
            st.error("Please upload or capture at least one clear face image.")
            return

        if total_samples > settings.max_upload_images:
            st.error(f"Select at most {settings.max_upload_images} images per registration.")
            return
        if total_samples < 3:
            st.info("You can add more views later; 3-5 views are recommended.")
        embeddings: list[np.ndarray] = []
        saved = 0
        rejected = 0

        def process_image(image: Image.Image, prefix: str, sample_label: str) -> None:
            nonlocal saved, rejected
            image = ImageOps.exif_transpose(image).convert("RGB")
            rgb = np.array(image)
            faces = face_service.detect_faces(rgb)
            valid, _, reason = face_service.evaluate_quality(rgb, faces)
            def reject(reason: str) -> None:
                nonlocal rejected
                rejected += 1
                st.warning(f"{sample_label}: {face_service.rejection_message(reason)}")
                logger.info("Face registration photo rejected: employee=%s reason=%s", employee_id, reason)
                with st.expander(f"View photo: {sample_label}"):
                    annotated = face_service.draw_annotations(rgb, faces) if faces else rgb
                    st.image(annotated, caption="Check face size, lighting, and focus", width=320)
            if not valid:
                reject(reason)
                return
            result = face_service.extract_face_embedding(rgb, faces)
            if result is None:
                reject("face_features_unavailable")
                return
            if embeddings and face_service.compare_embedding(result.embedding, embeddings)[0] < settings.threshold:
                reject("inconsistent_face")
                return
            sample_path = _save_image(selected_label.split(" - ", 1)[0], image, prefix)
            embeddings.append(result.embedding)
            with get_db_session() as session:
                session.add(
                    FaceEmbedding(
                        employee_id=employee_id,
                        model_name=face_service.model_name,
                        embedding=array_to_blob(result.embedding),
                        normalized_embedding=array_to_blob(face_service.normalize_embedding(result.embedding)),
                        embedding_dim=int(result.embedding.shape[0]),
                        source_image_path=str(sample_path),
                        quality_score=float(result.quality_score),
                        is_primary=False,
                    )
                )
            saved += 1

        for idx, file in enumerate(uploaded_files or []):
            try:
                with Image.open(file) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
            except (OSError, ValueError):
                rejected += 1
                st.warning(f"Could not read {file.name}. Upload a valid JPG or PNG photo.")
                continue
            process_image(image, f"upload_{idx}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}", file.name)

        if camera_photo is not None:
            image = Image.open(camera_photo)
            process_image(image, f"camera_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}", "Camera photo")

        if embeddings:
            avg_embedding = face_service.average_embedding(embeddings)
            with get_db_session() as session:
                session.add(
                    FaceEmbedding(
                        employee_id=employee_id,
                        model_name=face_service.model_name,
                        embedding=array_to_blob(avg_embedding),
                        normalized_embedding=array_to_blob(avg_embedding),
                        embedding_dim=int(avg_embedding.shape[0]),
                        source_image_path="average_embedding",
                        quality_score=1.0,
                        is_primary=True,
                    )
                )

        if saved:
            st.success(f"Added {saved} face photos. Rejected samples: {rejected}. Live recognition refreshes within 10 seconds.")
        else:
            st.error("No faces added. Retake the rejected photos and try again.")
        logger.info("Registered %s face embeddings for employee %s", saved, employee_id)
