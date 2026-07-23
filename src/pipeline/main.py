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
from pipeline.distance import build_distance_estimator
from pipeline.visualizer import Visualizer
from src.pipeline.tracking import KalmanPredictor #cambio


class SpeedrEyePipeline:
    def __init__(self, config, video_path=None, distance_method=None):
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
        self.distance_method = distance_method or config.DISTANCE_METHOD
        self.distance_estimator = build_distance_estimator(
            self.distance_method,
            self.detector,
            config,
        )
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

        #cambios
        current_frame_ids = set()

        for det in detections:
            track_id = det.get("track_id")
            
            if track_id is not None:
                current_frame_ids.add(track_id)
                x1, y1, x2, y2 = det["bbox"]
                
                # 1. Punto de contacto con el suelo (Pies del peatón/ciclista)
                feet_x = int((x1 + x2) / 2)
                feet_y = int(y2)

                # 2. Si es un nuevo ID, inicializamos su Filtro de Kalman
                if track_id not in self.trackers:
                    self.trackers[track_id] = KalmanPredictor()

                # 3. Le pasamos a Kalman la posición actual real (Medición + Actualización)
                self.trackers[track_id].update(feet_x, feet_y)

                # 4. Predecimos los próximos 30 cuadros en el futuro (~1 segundo a 30 FPS)
                future_path = self.trackers[track_id].predict_future(steps=30)
                
                # 5. Guardamos la trayectoria predicha dentro del diccionario de la detección
                det["future_path"] = future_path

        # 6. Limpieza: Eliminamos los Filtros de objetos que ya no están en pantalla
        lost_ids = set(self.trackers.keys()) - current_frame_ids
        for lost_id in lost_ids:
            del self.trackers[lost_id]
        # =========================================================================

        start_distance = time.perf_counter()
        if self.distance_estimator is not None:
            detections = self.distance_estimator.estimate(detections, frame.shape)
        distance_time = (time.perf_counter() - start_distance) * 1000

        start_visualization = time.perf_counter()
        # Ahora 'detections' lleva consigo la clave 'future_path' para que el visualizer la dibuje
        output = self.visualizer.draw(frame, detections)
        visualization_time = (time.perf_counter() - start_visualization) * 1000
        

        ##
        start_distance = time.perf_counter()
        if self.distance_estimator is not None:
            detections = self.distance_estimator.estimate(detections, frame.shape)
        distance_time = (time.perf_counter() - start_distance) * 1000

        start_visualization = time.perf_counter()
        output = self.visualizer.draw(frame, detections)
        visualization_time = (time.perf_counter() - start_visualization) * 1000

        total_time = (time.perf_counter() - start_total) * 1000
        self.fps_buffer.append(1000 / total_time if total_time > 0 else 0)
        self.frame_count += 1

        metrics = {
            "fps": np.mean(self.fps_buffer),
            "detections": len(detections),
            "detection_time": detection_time,
            "distance_time": distance_time,
            "distance_method": self.distance_method,
            "visualization_time": visualization_time,
            "total_time": total_time,
        }
        self.visualizer.draw_ui(output, metrics)
        return output, detections, metrics

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
    parser.add_argument(
        "--distance-method",
        choices=("none", "direct", "geometry"),
        default=Config.DISTANCE_METHOD,
    )
    args = parser.parse_args()

    source = 0 if args.camera or not args.video else args.video
    pipeline = SpeedrEyePipeline(
        Config,
        video_path=source,
        distance_method=args.distance_method,
    )
    pipeline.run_video(source)


if __name__ == "__main__":
    main()
