from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from config import settings
from utils.logger import logger


@dataclass(slots=True)
class FaceResult:
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    confidence: float
    quality_score: float
    face_image: np.ndarray
    reason: str = "accepted"
    kps: np.ndarray | None = None


class FaceService:
    def __init__(self) -> None:
        self.model = None
        self.recognizer = None
        self.yunet = None
        self.model_name = "unavailable"
        self.initialization_error = ""
        self.embedding_dim = 128
        self._max_frame_width = max(160, settings.max_frame_width)
        if settings.low_power_mode:
            cv2.setNumThreads(1)
        try:
            if settings.face_backend == "sface":
                if not settings.yunet_model_path.is_file() or not settings.sface_model_path.is_file():
                    raise FileNotFoundError("Run python scripts/download_face_models.py to install face models.")
                self.yunet = cv2.FaceDetectorYN.create(str(settings.yunet_model_path), "", (320, 320), settings.face_detection_confidence, 0.3, 500)
                self.recognizer = cv2.FaceRecognizerSF.create(str(settings.sface_model_path), "")
                self.model_name = "sface-2021dec-v1"
            elif settings.face_backend == "insightface":
                from insightface.app import FaceAnalysis
                self.model = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
                self.model.prepare(ctx_id=-1, det_size=(settings.insightface_det_size, settings.insightface_det_size))
                self.embedding_dim = 512
                self.model_name = "insightface-buffalo_l-rgb-v1"
            else:
                raise ValueError("FACE_BACKEND must be sface or insightface")
        except Exception as exc:
            self.model = self.recognizer = self.yunet = None
            self.initialization_error = str(exc)
            logger.warning("Face recognition unavailable: %s", exc)

    @property
    def recognition_available(self) -> bool:
        return self.recognizer is not None or self.model is not None

    @staticmethod
    def _ensure_rgb(image: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        # NumPy callers supply RGB, just like PIL and the live camera pipeline.
        return image

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_n = FaceService.normalize_embedding(a)
        b_n = FaceService.normalize_embedding(b)
        return float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))

    @staticmethod
    def is_blurry(image: np.ndarray, threshold: float = 80.0) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    @staticmethod
    def is_dark(image: np.ndarray, threshold: float = 55.0) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return float(np.mean(gray)) < threshold

    @staticmethod
    def _largest_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
        if not boxes:
            return None
        return max(boxes, key=lambda box: max(box[2] - box[0], 0) * max(box[3] - box[1], 0))

    def detect_faces(self, frame: np.ndarray | Image.Image) -> list[dict[str, Any]]:
        image = self._ensure_rgb(frame)
        original_image = image
        original_h, original_w = image.shape[:2]
        scale_x = scale_y = 1.0
        if original_w > self._max_frame_width:
            scale = self._max_frame_width / float(max(original_w, 1))
            resized_h = max(int(round(original_h * scale)), 1)
            image = cv2.resize(image, (self._max_frame_width, resized_h), interpolation=cv2.INTER_AREA)
            scale_x = original_w / float(self._max_frame_width)
            scale_y = original_h / float(resized_h)
        faces: list[dict[str, Any]] = []

        if self.yunet is not None:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.yunet.setInputSize((image.shape[1], image.shape[0]))
            _, detections = self.yunet.detect(bgr)
            for detection in detections if detections is not None else []:
                x, y, w, h = detection[:4]
                bbox = (max(0, int(x * scale_x)), max(0, int(y * scale_y)),
                        min(original_w, int((x + w) * scale_x)), min(original_h, int((y + h) * scale_y)))
                x1, y1, x2, y2 = bbox
                crop = original_image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # Skip feature extraction for faces too small, dark or blurred to match reliably.
                embedding = np.zeros(self.embedding_dim, dtype=np.float32)
                if min(x2 - x1, y2 - y1) >= settings.face_detector_min_size and not self.is_dark(crop) and not self.is_blurry(crop):
                    aligned = self.recognizer.alignCrop(bgr, detection)
                    embedding = self.normalize_embedding(self.recognizer.feature(aligned))
                kps = detection[4:14].reshape(5, 2) * np.array([scale_x, scale_y])
                faces.append({"bbox": bbox, "embedding": embedding, "confidence": float(detection[-1]), "kps": kps})
            return faces

        if self.model is not None:
            try:
                results = self.model.get(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                for face in results:
                    x1, y1, x2, y2 = map(int, face.bbox.tolist())
                    if scale_x != 1.0 or scale_y != 1.0:
                        x1 = int(round(x1 * scale_x))
                        x2 = int(round(x2 * scale_x))
                        y1 = int(round(y1 * scale_y))
                        y2 = int(round(y2 * scale_y))
                    faces.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "embedding": self.normalize_embedding(np.asarray(face.embedding, dtype=np.float32)),
                            "confidence": float(getattr(face, "det_score", 1.0)),
                            "kps": np.asarray(face.kps) * np.array([scale_x, scale_y]) if getattr(face, "kps", None) is not None else None,
                        }
                    )
                return faces
            except Exception as exc:  # pragma: no cover - runtime dependent
                logger.warning("InsightFace detection failed: %s", exc)
                return []

        # No substitute pixel embeddings: unavailable models cannot identify people.
        return []

    def evaluate_quality(self, image: np.ndarray, faces: list[dict[str, Any]] | None = None) -> tuple[bool, float, str]:
        if faces is None:
            faces = self.detect_faces(image)
        if len(faces) == 0:
            return False, 0.0, "no_face_detected"
        if len(faces) > 1:
            return False, 0.0, "multiple_faces"

        face = faces[0]
        x1, y1, x2, y2 = face["bbox"]
        h, w = image.shape[:2]
        area_ratio = max(0.0, ((x2 - x1) * (y2 - y1)) / float(max(w * h, 1)))
        if area_ratio < 0.04:
            return False, 0.2, "face_too_small"
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0 or min(crop.shape[:2]) < settings.face_detector_min_size:
            return False, 0.0, "face_too_small"
        if self.is_dark(crop):
            return False, 0.0, "too_dark"
        if self.is_blurry(crop):
            return False, 0.0, "blurred"
        if not self.recognition_available or not np.any(face["embedding"]):
            return False, 0.0, "recognition_unavailable"
        quality = float(np.clip((area_ratio * 8.0) + face["confidence"] * 0.2, 0.0, 1.0))
        return True, quality, "accepted"

    def extract_face_embedding(self, image: np.ndarray | Image.Image, faces: list[dict[str, Any]] | None = None) -> FaceResult | None:
        rgb = self._ensure_rgb(image)
        if faces is None:
            faces = self.detect_faces(rgb)
        is_valid, quality, reason = self.evaluate_quality(rgb, faces)
        if not is_valid:
            return None
        if not faces:
            return None
        face = faces[0]
        x1, y1, x2, y2 = face["bbox"]
        crop = rgb[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        embedding = np.asarray(face["embedding"], dtype=np.float32).reshape(-1)
        if embedding.size != self.embedding_dim or not np.all(np.isfinite(embedding)):
            return None
        embedding = self.normalize_embedding(embedding)
        return FaceResult(
            bbox=face["bbox"],
            embedding=embedding,
            confidence=float(face["confidence"]),
            quality_score=quality,
            face_image=crop,
            reason=reason,
            kps=face.get("kps"),
        )

    def average_embedding(self, embeddings: list[np.ndarray]) -> np.ndarray:
        if not embeddings:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        stacked = np.stack([self.normalize_embedding(e) for e in embeddings], axis=0)
        return self.normalize_embedding(np.mean(stacked, axis=0))

    def compare_embedding(self, probe: np.ndarray, candidates: list[np.ndarray]) -> tuple[float, int]:
        if not candidates:
            return 0.0, -1
        probe_n = self.normalize_embedding(probe)
        scores = [float(np.dot(probe_n, self.normalize_embedding(candidate))) for candidate in candidates]
        best_index = int(np.argmax(scores))
        return float(scores[best_index]), best_index

    def draw_annotations(
        self,
        frame: np.ndarray,
        faces: list[dict[str, Any]],
        labels: list[str] | None = None,
    ) -> np.ndarray:
        image = frame.copy()
        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face["bbox"])
            label = labels[idx] if labels and idx < len(labels) else "Unknown"
            color = (34, 197, 94) if label != "Unknown" else (239, 68, 68)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(image, (x1, max(0, y1 - 28)), (x1 + 220, y1), color, -1)
            cv2.putText(
                image,
                label,
                (x1 + 8, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return image

    def load_image_bytes(self, file_bytes: bytes) -> np.ndarray:
        array = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
