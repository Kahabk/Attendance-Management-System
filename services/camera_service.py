from __future__ import annotations

import time
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
    def __init__(self, threshold: float | None = None) -> None:
        self.face_service = FaceService()
        self.threshold = threshold or attendance_service.recognition_threshold()
        self._cache: dict[str, Any] = {"timestamp": 0.0, "embeddings": []}

    def _refresh_cache(self) -> None:
        now = time.time()
        if now - float(self._cache["timestamp"]) < 10.0 and self._cache["embeddings"]:
            return

        records: list[dict[str, Any]] = []
        with get_db_session() as session:
            rows = (
                session.query(FaceEmbedding, Employee)
                .join(Employee, Employee.id == FaceEmbedding.employee_id)
                .options(joinedload(Employee.department))
                .filter(Employee.is_active.is_(True))
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
        self._cache = {"timestamp": now, "embeddings": records}

    def recognize(self, frame_rgb: np.ndarray, camera_id: str = "default", device: str = "webcam") -> tuple[np.ndarray, list[RecognitionHit]]:
        self._refresh_cache()
        faces = self.face_service.detect_faces(frame_rgb)
        labels: list[str] = []
        hits: list[RecognitionHit] = []
        candidate_embeddings = [row["embedding"] for row in self._cache["embeddings"]]

        for face in faces:
            embedding = np.asarray(face["embedding"], dtype=np.float32)
            bbox = tuple(int(v) for v in face["bbox"])
            if not candidate_embeddings:
                labels.append("Unknown")
                continue
            best_score = -1.0
            best_index = -1
            probe = self.face_service.normalize_embedding(embedding)
            for idx, candidate in enumerate(candidate_embeddings):
                score = float(np.dot(probe, self.face_service.normalize_embedding(candidate)))
                if score > best_score:
                    best_score = score
                    best_index = idx
            if best_index < 0 or best_score < self.threshold:
                labels.append("Unknown")
                continue

            matched = self._cache["embeddings"][best_index]
            labels.append(f'{matched["employee_code"]} - {matched["employee_name"]} ({best_score:.2f})')
            hits.append(
                RecognitionHit(
                    face_embedding_id=int(matched["face_embedding_id"]),
                    employee_id=int(matched["employee_id"]),
                    employee_name=str(matched["employee_name"]),
                    employee_code=str(matched["employee_code"]),
                    department=str(matched["department"] or ""),
                    similarity=float(best_score),
                    bbox=bbox,
                    status="matched",
                    face_sample_path=str(matched["face_sample_path"] or ""),
                )
            )
            try:
                marked, message = attendance_service.record_recognition(
                    employee_id=int(matched["employee_id"]),
                    confidence=float(best_score),
                    face_embedding_id=int(matched["face_embedding_id"]),
                    camera_id=camera_id,
                    device=device,
                )
                logger.info("Attendance recognition result for %s: %s", matched["employee_code"], message)
                hits[-1].status = "marked" if marked else "ignored"
                hits[-1].attendance_message = message
            except Exception as exc:  # pragma: no cover - real-time runtime path
                logger.warning("Attendance marking failed during recognition: %s", exc)

        annotated = self.face_service.draw_annotations(frame_rgb, faces, labels)
        return annotated, hits


class LiveVideoProcessor(VideoProcessorBase):  # pragma: no cover - depends on runtime websocket loop
    def __init__(self) -> None:
        self.engine = RecognitionEngine()
        self.last_hits: list[RecognitionHit] = []
        self.fps = 0.0
        self._last_frame_time = time.time()
        self._last_inference_time = 0.0
        self._last_annotated_frame: np.ndarray | None = None
        self._min_interval = max(settings.live_recognition_interval_ms / 1000.0, 0.1)

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        now = time.time()
        start = now
        if self._last_annotated_frame is None or now - self._last_inference_time >= self._min_interval:
            annotated, hits = self.engine.recognize(rgb)
            self.last_hits = hits
            self._last_annotated_frame = annotated
            self._last_inference_time = now
            elapsed = max(time.time() - start, 1e-6)
            self.fps = 1.0 / elapsed
        else:
            annotated = self._last_annotated_frame
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"FPS: {self.fps:.1f}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(bgr, format="bgr24")
