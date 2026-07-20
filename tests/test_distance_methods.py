"""Small deterministic tests for the two distance methods."""

import unittest
from collections import deque
from pathlib import Path

import numpy as np
import torch
import cv2

from src.pipeline.main import SpeedrEyePipeline
from src.pipeline.visualizer import Visualizer
from src.pipeline.distance.geometry_guided import GeometryGuidedDistanceEstimator
from src.pipeline.distance.head import DistanceRegressionHead, prepare_distance_inputs


class TestConfig:
    GEOMETRY_DISTANCE_WEIGHTS = Path("missing-geometry-weights.pt")
    CLASS_HEIGHTS = {0: 1.70, 1: 1.70}
    FOCAL_LENGTH = 800.0
    MIN_DISTANCE = 0.5
    MAX_DISTANCE = 60.0
    CLASS_COLORS = {0: (0, 255, 0), 1: (0, 165, 255)}
    CLASS_NAMES = {0: "Peaton", 1: "Ciclista"}
    BOX_THICKNESS = 1
    TEXT_SCALE = 0.4
    TEXT_THICKNESS = 1
    UI_TEXT_COLOR = (0, 255, 255)
    FONT = cv2.FONT_HERSHEY_SIMPLEX


class DistanceMethodTests(unittest.TestCase):
    def test_geometry_distance(self):
        estimator = GeometryGuidedDistanceEstimator(detector=None, config=TestConfig)
        detections = [{"bbox": (20, 10, 80, 210), "class": 0, "conf": 0.9}]

        result = estimator.estimate(detections, (240, 320, 3))

        self.assertAlmostEqual(result[0]["distance"], 6.8)
        self.assertEqual(result[0]["distance_method"], "geometry")

    def test_head_accepts_mapped_rois(self):
        feature_map = torch.rand(1, 16, 80, 80)
        detections = [
            {"bbox": (100, 50, 300, 400)},
            {"bbox": (400, 120, 600, 460)},
        ]
        rois, box_features = prepare_distance_inputs(
            detections,
            original_shape=(480, 640, 3),
            feature_map=feature_map,
            stride=8,
        )
        head = DistanceRegressionHead(feature_channels=16)

        prediction = head(feature_map, rois, box_features)

        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_pipeline_records_distance_and_complete_timings(self):
        class FakeDetector:
            def detect(self, _frame):
                return [{"bbox": (20, 20, 80, 120), "class": 0, "conf": 0.9}]

        class FakeCalibrator:
            is_calibrated = False
            horizon_line = None

        pipeline = SpeedrEyePipeline.__new__(SpeedrEyePipeline)
        pipeline.config = TestConfig
        pipeline.fps_buffer = deque(maxlen=30)
        pipeline.frame_count = 0
        pipeline.calibrator = FakeCalibrator()
        pipeline.detector = FakeDetector()
        pipeline.distance_method = "geometry"
        pipeline.distance_estimator = GeometryGuidedDistanceEstimator(None, TestConfig)
        pipeline.visualizer = Visualizer(TestConfig)

        output, detections, metrics = pipeline.process_frame(
            np.zeros((160, 160, 3), dtype=np.uint8)
        )

        self.assertEqual(output.shape, (160, 160, 3))
        self.assertGreater(detections[0]["distance"], 0)
        self.assertEqual(
            set(metrics),
            {
                "fps",
                "detections",
                "detection_time",
                "distance_time",
                "distance_method",
                "visualization_time",
                "total_time",
            },
        )


if __name__ == "__main__":
    unittest.main()
