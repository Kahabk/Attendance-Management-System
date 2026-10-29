from __future__ import annotations

import unittest
import time
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from sqlalchemy import create_engine, text

from config import settings
from models import Base
from database_migrations import upgrade_database
from services.face_service import FaceService
from services.camera_service import RecognitionEngine, LiveVideoProcessor
from services.face_matching import select_match, ConfirmationTracker
from utils.logger import logger


def setUpModule():
    logger.disabled = True


def tearDownModule():
    logger.disabled = False


class MatchingTests(unittest.TestCase):
    def test_rejects_ambiguous_people(self):
        self.assertEqual(select_match(np.array([0.8, 0.76]), [1, 2], .45, .08), -1)

    def test_multiple_photos_same_person_are_not_rivals(self):
        self.assertEqual(select_match(np.array([0.8, 0.79, 0.5]), [1, 1, 2], .45, .08), 0)

    def test_rejects_unknown_and_corrupt_scores(self):
        for scores in ([.3, .2], [float("nan"), .9], [float("inf"), .2]):
            self.assertEqual(select_match(np.array(scores), [1, 2], .45, .08), -1)
        self.assertEqual(select_match(np.array([]), [], .45, .08), -1)

    def test_confirmation_requires_consecutive_frames(self):
        tracker = ConfirmationTracker(3)
        for expected in (False, False, True):
            tracker.begin_frame()
            self.assertEqual(tracker.observe(1), expected)
            tracker.end_frame()
        tracker.begin_frame()
        tracker.end_frame()
        tracker.begin_frame()
        self.assertFalse(tracker.observe(1))

    def test_duplicate_face_in_frame_does_not_count_twice(self):
        tracker = ConfirmationTracker(2)
        tracker.begin_frame()
        self.assertFalse(tracker.observe(1))
        self.assertFalse(tracker.observe(1))
        tracker.end_frame()
        tracker.begin_frame()
        self.assertTrue(tracker.observe(1))


class ServiceTests(unittest.TestCase):
    def test_missing_models_never_generate_identity_embeddings(self):
        with patch("pathlib.Path.is_file", return_value=False):
            service = FaceService()
        self.assertFalse(service.recognition_available)
        self.assertIsNone(service.extract_face_embedding(np.zeros((200, 200, 3), dtype=np.uint8)))

    def test_rgb_input_is_preserved(self):
        image = np.array([[[255, 0, 20]]], dtype=np.uint8)
        np.testing.assert_array_equal(FaceService._ensure_rgb(image), image)

    @unittest.skipUnless(settings.yunet_model_path.exists() and settings.sface_model_path.exists(), "Models not installed")
    def test_actual_models_load_and_blank_frame_has_no_faces(self):
        service = FaceService()
        self.assertTrue(service.recognition_available, service.initialization_error)
        self.assertEqual(service.embedding_dim, 128)
        self.assertEqual(service.detect_faces(np.zeros((480, 640, 3), dtype=np.uint8)), [])

    def test_single_detection_is_reused_for_registration(self):
        service = FaceService.__new__(FaceService)
        service.embedding_dim = 128
        service.model = None
        face = {"bbox": (10, 10, 90, 90), "embedding": np.ones(128), "confidence": .99}
        image = np.ones((100, 100, 3), dtype=np.uint8)
        with patch.object(service, "detect_faces", return_value=[face]) as detect, patch.object(service, "evaluate_quality", return_value=(True, 1., "accepted")):
            result = service.extract_face_embedding(image)
        self.assertIsNotNone(result)
        detect.assert_called_once()


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.service = FaceService.__new__(FaceService)
        self.service.recognizer = object()
        self.service.model = None
        rng = np.random.default_rng(10)
        self.crop = rng.integers(70, 190, (112, 112, 3), dtype=np.uint8)
        self.face = dict(bbox=(0, 0, 112, 112), embedding=np.ones(128), confidence=.99)

    def test_valid_face_in_large_background_is_not_rejected_by_area(self):
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        image[:112, :112] = self.crop
        self.assertTrue(self.service.evaluate_quality(image, [self.face])[0])

    def test_blur_measurement_is_consistent_across_resolutions(self):
        large = np.repeat(np.repeat(self.crop, 4, axis=0), 4, axis=1)
        small_metrics = self.service.quality_metrics(self.crop)
        large_metrics = self.service.quality_metrics(large)
        self.assertAlmostEqual(small_metrics["sharpness"], large_metrics["sharpness"], places=3)

    def test_severely_blurred_face_is_still_rejected(self):
        image = np.full((112, 112, 3), 120, dtype=np.uint8)
        self.assertEqual(self.service.evaluate_quality(image, [self.face])[2], "blurred")

    def test_dark_face_is_still_rejected(self):
        self.assertEqual(self.service.evaluate_quality(self.crop // 5, [self.face])[2], "too_dark")

    def test_tiny_face_is_still_rejected(self):
        self.face["bbox"] = (0, 0, 20, 20)
        self.assertEqual(self.service.evaluate_quality(self.crop, [self.face])[2], "face_too_small")

    def test_invalid_features_have_distinct_error_from_missing_model(self):
        self.face["embedding"] = np.zeros(128)
        self.assertEqual(self.service.evaluate_quality(self.crop, [self.face])[2], "face_features_unavailable")
        self.service.recognizer = None
        self.assertEqual(self.service.evaluate_quality(self.crop, [self.face])[2], "recognition_unavailable")


class MigrationTests(unittest.TestCase):
    def test_legacy_records_are_preserved_and_tagged(self):
        engine = create_engine("sqlite://")
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE face_embeddings (id INTEGER PRIMARY KEY)"))
            connection.execute(text("INSERT INTO face_embeddings (id) VALUES (1)"))
        upgrade_database(engine)
        upgrade_database(engine)
        with engine.connect() as connection:
            self.assertEqual(connection.execute(text("SELECT model_name FROM face_embeddings WHERE id=1")).scalar(), "legacy")
        engine.dispose()


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = RecognitionEngine.__new__(RecognitionEngine)
        self.engine.face_service = FaceService.__new__(FaceService)
        self.engine.threshold = .45
        self.engine.confirmations = ConfirmationTracker(2)
        self.engine._last_seen = {}
        self.engine._last_status = {}
        self.vector = np.zeros(128, dtype=np.float32)
        self.vector[0] = 1.
        record = dict(employee_id=1, employee_code="E1", employee_name="Test", department="", face_embedding_id=1, face_sample_path="test.png")
        self.engine._cache = dict(timestamp=time.monotonic(), embeddings=[record], matrix=self.vector.reshape(1, -1))
        self.frame = np.zeros((160, 160, 3), dtype=np.uint8)
        self.face = dict(bbox=(0, 0, 100, 100), embedding=self.vector, confidence=.99)

    def test_marks_only_after_confirmation_and_preserves_checkout_status(self):
        with patch.object(self.engine.face_service, "detect_faces", return_value=[self.face]), patch.object(self.engine.face_service, "evaluate_quality", return_value=(True, 1., "accepted")), patch("services.camera_service.attendance_service.record_recognition", return_value=(True, "Check-out recorded")) as mark:
            self.assertEqual(self.engine.recognize(self.frame)[1], [])
            self.assertEqual(self.engine.recognize(self.frame)[1][0].status, "OUT")
            self.assertEqual(self.engine.recognize(self.frame)[1][0].status, "OUT")
            mark.assert_called_once()

    def test_empty_cache_does_not_query_each_frame(self):
        self.engine._cache = dict(timestamp=time.monotonic(), embeddings=[], matrix=None)
        with patch("services.camera_service.get_db_session") as session:
            self.engine._refresh_cache()
            session.assert_not_called()

    def test_zero_embedding_cannot_mark_attendance(self):
        self.face["embedding"] = np.zeros(128, dtype=np.float32)
        with patch.object(self.engine.face_service, "detect_faces", return_value=[self.face]), patch("services.camera_service.attendance_service.record_recognition") as mark:
            self.assertEqual(self.engine.recognize(self.frame)[1], [])
            mark.assert_not_called()

    def test_throttled_preview_uses_current_frame(self):
        processor = LiveVideoProcessor.__new__(LiveVideoProcessor)
        processor._lock = Lock()
        processor._last_inference_time = time.monotonic()
        processor._min_interval = 10.
        processor.fps = 0.
        processor.engine = SimpleNamespace(recognize=None)
        image = np.full((160, 160, 3), 77, dtype=np.uint8)
        frame = SimpleNamespace(to_ndarray=lambda **kwargs: image)
        fake_av = SimpleNamespace(VideoFrame=SimpleNamespace(from_ndarray=lambda array, **kwargs: array))
        with patch("services.camera_service.av", fake_av):
            result = processor.recv(frame)
        np.testing.assert_array_equal(result[159, 159], [77, 77, 77])

    def test_cache_only_queries_current_model(self):
        from sqlalchemy.orm import Session
        from models import Employee, FaceEmbedding
        from contextlib import contextmanager
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            employee = Employee(employee_id="E1", name="Test", is_active=True)
            session.add(employee)
            session.flush()
            for name in ("legacy", "sface-2021dec-v1"):
                session.add(FaceEmbedding(employee_id=employee.id, embedding=self.vector.tobytes(), normalized_embedding=self.vector.tobytes(), embedding_dim=128, model_name=name))
            session.commit()
        @contextmanager
        def test_session():
            with Session(engine) as session:
                yield session
        self.engine.face_service.model_name = "sface-2021dec-v1"
        self.engine.face_service.embedding_dim = 128
        self.engine._cache["timestamp"] = None
        with patch("services.camera_service.get_db_session", test_session):
            self.engine._refresh_cache()
        self.assertEqual(len(self.engine._cache["embeddings"]), 1)
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
