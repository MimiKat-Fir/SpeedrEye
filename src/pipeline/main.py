#!/usr/bin/env python3
"""SpeedrEye detection pipeline."""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.calibration import CameraCalibrator
from pipeline.config import Config
from pipeline.detector import Detector
from pipeline.visualizer import Visualizer


class SpeedrEyePipeline:
    def __init__(self, config, video_path=None):
        self.config = config
        self.fps_buffer = deque(maxlen=config.FPS_BUFFER_SIZE)
        self.frame_count = 0

        self.calibrator = CameraCalibrator(config)
        if video_path and Path(video_path).exists():
            self.calibrator.calibrate_from_video(video_path, num_frames=20)

        params = self.calibrator.get_parameters()
        config.FOCAL_LENGTH = params["focal_length"]
        config.CX = params["cx"]
        config.CY = params["cy"]

        self.detector = Detector(config)
        self.visualizer = Visualizer(config)

    def process_frame(self, frame):
        start_total = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_detection = time.perf_counter()
        detections = self.detector.detect(frame_rgb)
        detection_time = (time.perf_counter() - start_detection) * 1000

        if self.frame_count % 10 == 0 and self.calibrator.is_calibrated:
            self.calibrator.update_scene_params(frame)

        if self.calibrator.horizon_line is not None:
            horizon_y = self.calibrator.horizon_line
            detections = [
                detection
                for detection in detections
                if (detection["bbox"][1] + detection["bbox"][3]) / 2 > horizon_y
            ]

        total_time = (time.perf_counter() - start_total) * 1000
        self.fps_buffer.append(1000 / total_time if total_time > 0 else 0)
        self.frame_count += 1

        metrics = {
            "fps": np.mean(self.fps_buffer),
            "detections": len(detections),
            "detection_time": detection_time,
            "total_time": total_time,
        }
        return self.visualizer.draw(frame, detections, metrics), detections, metrics

    def run_video(self, video_path):
        if isinstance(video_path, str) and not Path(video_path).exists():
            print(f"No se encuentra: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("No se pudo abrir la fuente")
            return

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            output, _, _ = self.process_frame(frame)
            cv2.imshow("SpeedrEye", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SpeedrEye object detection")
    parser.add_argument("--video", type=str, help="Ruta al video")
    parser.add_argument("--camera", action="store_true", help="Usar camara")
    args = parser.parse_args()

    source = 0 if args.camera or not args.video else args.video
    pipeline = SpeedrEyePipeline(Config, video_path=source)
    pipeline.run_video(source)


if __name__ == "__main__":
    main()
