from __future__ import annotations

import time
from threading import Lock
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from sqlalchemy.orm import joinedload

from config import settings
from database import get_db_session
from models import Employee, FaceEmbedding
from services.attendance_service import attendance_service
from services.face_service import FaceService
from services.face_matching import ConfirmationTracker, select_match
from utils.logger import logger
from utils.serialization import blob_to_array

try:  # pragma: no cover - optional UI dependency
    import av
    from streamlit_webrtc import VideoProcessorBase
except Exception:  # pragma: no cover
    av = None
    VideoProcessorBase = object


@dataclass(slots=True)
class RecognitionHit:
    employee_id: int
    employee_name: str
    employee_code: str
    department: str
    similarity: float
    bbox: tuple[int, int, int, int]
    status: str
    face_embedding_id: int | None = None
    face_sample_path: str | None = None
    attendance_message: str | None = None


class RecognitionEngine:
    def __init__(self, threshold: float | None = None, confirmations: int = 1) -> None:
        self.face_service = FaceService()
        self.threshold = attendance_service.recognition_threshold() if threshold is None else threshold
        self.confirmations = ConfirmationTracker(confirmations)
        self._cache: dict[str, Any] = {"timestamp": None, "embeddings": [], "matrix": None}
        self._last_seen: dict[int, float] = {}
        self._last_status: dict[int, tuple[str, str]] = {}

    def _refresh_cache(self) -> None:
        now = time.monotonic()
        if self._cache["timestamp"] is not None and now - self._cache["timestamp"] < 10.0:
            return

        records: list[dict[str, Any]] = []
        with get_db_session() as session:
            rows = (
                session.query(FaceEmbedding, Employee)
                .join(Employee, Employee.id == FaceEmbedding.employee_id)
                .options(joinedload(Employee.department))
                .filter(Employee.is_active.is_(True), FaceEmbedding.model_name == self.face_service.model_name)
                .all()
            )
            for embedding_row, employee in rows:
                records.append(
                    {
                        "face_embedding_id": embedding_row.id,
                        "employee_id": employee.id,
                        "employee_name": employee.name,
                        "employee_code": employee.employee_id,
                        "department": employee.department.name if employee.department else "",
                        "face_sample_path": embedding_row.source_image_path,
                        "embedding": blob_to_array(embedding_row.normalized_embedding),
                    }
                )
        valid_records = []
        vectors = []
        for record in records:
            vector = np.asarray(record.pop("embedding"), dtype=np.float32).reshape(-1)
            if vector.size != self.face_service.embedding_dim or not np.all(np.isfinite(vector)):
                continue
            norm = np.linalg.norm(vector)
            if norm <= 0:
                continue
            valid_records.append(record)
            vectors.append(vector / norm)
        matrix = np.ascontiguousarray(vectors, dtype=np.float32) if vectors else None
        self._cache = {"timestamp": now, "embeddings": valid_records, "matrix": matrix}

    def recognize(self, frame_rgb: np.ndarray, camera_id: str = "default", device: str = "webcam") -> tuple[np.ndarray, list[RecognitionHit]]:
        self._refresh_cache()
        faces = self.face_service.detect_faces(frame_rgb)
        labels: list[str] = []
        hits: list[RecognitionHit] = []
        candidate_matrix = self._cache["matrix"]

        confirmed_this_frame: set[int] = set()
        self.confirmations.begin_frame()
        for face in faces:
            embedding = np.asarray(face["embedding"], dtype=np.float32)
            bbox = tuple(int(v) for v in face["bbox"])
            if candidate_matrix is None or not np.any(embedding):
                labels.append("Unknown")
                continue
            valid, _, _ = self.face_service.evaluate_quality(frame_rgb, [face])
            if not valid:
                labels.append("Unknown")
                continue
            probe = self.face_service.normalize_embedding(embedding)
            if probe.size != candidate_matrix.shape[1] or not np.all(np.isfinite(probe)):
                labels.append("Unknown")
                continue
            scores = candidate_matrix @ probe
            best_index = select_match(scores, [row["employee_id"] for row in self._cache["embeddings"]], self.threshold, settings.recognition_margin)
            if best_index < 0:
                labels.append("Unknown")
                continue
            best_score = float(scores[best_index])
            matched = self._cache["embeddings"][best_index]
            employee_id = int(matched["employee_id"])
            if employee_id in confirmed_this_frame:
                labels.append("Unknown")
                continue
            confirmed_this_frame.add(employee_id)
            if not self.confirmations.observe(employee_id):
                labels.append("Confirming face...")
                continue
            labels.append(f'{matched["employee_code"]} - {matched["employee_name"]} ({best_score:.2f})')
            hits.append(
                RecognitionHit(
                    face_embedding_id=int(matched["face_embedding_id"]),
                    employee_id=employee_id,
                    employee_name=str(matched["employee_name"]),
                    employee_code=str(matched["employee_code"]),
                    department=str(matched["department"] or ""),
                    similarity=float(best_score),
                    bbox=bbox,
                    status="matched",
                    face_sample_path=str(matched["face_sample_path"] or ""),
                )
            )
            now_ts = time.monotonic()
            last_seen = self._last_seen.get(employee_id)
            should_mark = last_seen is None or (now_ts - last_seen) >= settings.attendance_reentry_gap_seconds

            if should_mark:
                try:
                    marked, message = attendance_service.record_recognition(
                        employee_id=employee_id,
                        confidence=float(best_score),
                        face_embedding_id=int(matched["face_embedding_id"]),
                        camera_id=camera_id,
                        device=device,
                    )
                    logger.info("Attendance recognition result for %s: %s", matched["employee_code"], message)
                    if marked and "Check-in" in message:
                        hits[-1].status = "IN"
                        labels[-1] = f'{matched["employee_code"]} - {matched["employee_name"]} IN ({best_score:.2f})'
                    elif marked and "Check-out" in message:
                        hits[-1].status = "OUT"
                        labels[-1] = f'{matched["employee_code"]} - {matched["employee_name"]} OUT ({best_score:.2f})'
                    else:
                        hits[-1].status = "DONE"
                        labels[-1] = f'{matched["employee_code"]} - {matched["employee_name"]} DONE ({best_score:.2f})'
                    hits[-1].attendance_message = message
                    self._last_seen[employee_id] = now_ts
                    self._last_status[employee_id] = (hits[-1].status, message)
                except Exception as exc:  # pragma: no cover - real-time runtime path
                    hits[-1].status = "ERROR"
                    hits[-1].attendance_message = "Attendance could not be saved; retrying."
                    logger.warning("Attendance marking failed during recognition: %s", exc)
            else:
                status, message = self._last_status.get(employee_id, ("matched", "Face matched"))
                hits[-1].status = status
                hits[-1].attendance_message = message
                self._last_seen[employee_id] = now_ts
                labels[-1] = f'{matched["employee_code"]} - {matched["employee_name"]} {status} ({best_score:.2f})'

        self.confirmations.end_frame()
        annotated = self.face_service.draw_annotations(frame_rgb, faces, labels)
        return annotated, hits


class LiveVideoProcessor(VideoProcessorBase):  # pragma: no cover - depends on runtime websocket loop
    def __init__(self) -> None:
        self.engine = RecognitionEngine(confirmations=settings.recognition_confirmations)
        self.last_hits: list[RecognitionHit] = []
        self.fps = 0.0
        self._last_inference_time = 0.0
        self._lock = Lock()
        self._min_interval = max(settings.live_recognition_interval_ms / 1000.0, 0.1)

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        bgr = img
        # A busy inference must not queue more work or freeze the camera preview.
        if now - self._last_inference_time >= self._min_interval and self._lock.acquire(blocking=False):
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                start = time.monotonic()
                annotated, hits = self.engine.recognize(rgb)
                self.last_hits = hits
                bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                self.fps = 1.0 / max(time.monotonic() - start, 1e-6)
                self._last_inference_time = time.monotonic()
            finally:
                self._lock.release()
        cv2.putText(bgr, f"Inference FPS: {self.fps:.1f}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(bgr, format="bgr24")
