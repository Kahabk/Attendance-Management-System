from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

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
    face_service = FaceService()

    with get_db_session() as session:
        employees = session.query(Employee).order_by(Employee.name.asc()).all()
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
        help="Upload front-facing, well-lit images. One clear face photo is enough.",
    )
    camera_photo = st.camera_input("Capture a face photo")

    total_samples = len(uploaded_files or []) + (1 if camera_photo else 0)
    st.write(f"Samples selected: {total_samples}")

    if st.button("Register Faces", width="stretch"):
        if total_samples < 1:
            st.error("Please upload or capture at least one clear face image.")
            return

        embeddings: list[np.ndarray] = []
        saved = 0
        rejected = 0
        sample_paths: list[str] = []

        def process_image(image: Image.Image, prefix: str) -> None:
            nonlocal saved, rejected
            rgb = np.array(image.convert("RGB"))
            result = face_service.extract_face_embedding(rgb)
            if result is None:
                rejected += 1
                return
            sample_path = _save_image(selected_label.split(" - ", 1)[0], image, prefix)
            sample_paths.append(str(sample_path))
            embeddings.append(result.embedding)
            with get_db_session() as session:
                session.add(
                    FaceEmbedding(
                        employee_id=employee_id,
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
            image = Image.open(file)
            process_image(image, f"upload_{idx}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

        if camera_photo is not None:
            image = Image.open(camera_photo)
            process_image(image, f"camera_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

        if embeddings:
            avg_embedding = face_service.average_embedding(embeddings)
            with get_db_session() as session:
                session.add(
                    FaceEmbedding(
                        employee_id=employee_id,
                        embedding=array_to_blob(avg_embedding),
                        normalized_embedding=array_to_blob(avg_embedding),
                        embedding_dim=int(avg_embedding.shape[0]),
                        source_image_path="average_embedding",
                        quality_score=1.0,
                        is_primary=True,
                    )
                )

        st.success(f"Registered {saved} embeddings. Rejected samples: {rejected}")
        logger.info("Registered %s face embeddings for employee %s", saved, employee_id)
