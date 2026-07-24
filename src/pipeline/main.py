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
from pipeline.tracking import KalmanPredictor #cambio
from pipeline.tracking.pose_estimator import PoseOrientationEstimator

class SpeedrEyePipeline:
    def __init__(self, config, video_path=None, distance_method=None):
        self.config = config
        self.fps_buffer = deque(maxlen=config.FPS_BUFFER_SIZE)
        self.frame_count = 0

        ##
        self.trackers = {}

        self.calibrator = CameraCalibrator(config)
        if video_path and Path(video_path).exists():
            self.calibrator.calibrate_from_video(video_path, num_frames=20)

        self.detector = Detector(config)
        # Forzamos a limpiar la memoria interna del tracker de Ultralytics
        if hasattr(self.detector.model, 'predictor') and self.detector.model.predictor is not None:
            self.detector.model.predictor.trackers = []

        params = self.calibrator.get_parameters()
        config.FOCAL_LENGTH = params["focal_length"]
        config.CX = params["cx"]
        config.CY = params["cy"]

        self.distance_method = distance_method or config.DISTANCE_METHOD
        self.distance_estimator = build_distance_estimator(
            self.distance_method,
            self.detector,
            config,
        )
        self.visualizer = Visualizer(config)

        self.pose_estimator = PoseOrientationEstimator()


    def process_frame(self, frame):
        start_total = time.perf_counter()

        # 1. Detección y seguimiento con YOLO
        start_detection = time.perf_counter()
        detections = self.detector.detect(frame)
        detection_time = (time.perf_counter() - start_detection) * 1000


        # 2. Recalibración periódica de la escena
        if self.frame_count % 10 == 0 and self.calibrator.is_calibrated:
            self.calibrator.update_scene_params(frame)

        # 3. Filtrado por línea del horizonte
        if self.calibrator.horizon_line is not None:
            horizon_y = self.calibrator.horizon_line
            detections = [
                detection
                for detection in detections
                if (detection["bbox"][1] + detection["bbox"][3]) / 2 > horizon_y
            ]

        # 4. Estimación de distancia
        start_distance = time.perf_counter()
        if self.distance_estimator is not None:
            detections = self.distance_estimator.estimate(detections, frame.shape)
        distance_time = (time.perf_counter() - start_distance) * 1000


        #kalman
        current_frame_ids = set()

        fx = getattr(self.config, "FOCAL_LENGTH", 800.0)
        cx = getattr(self.config, "CX", frame.shape[1] / 2)
        cy = getattr(self.config, "CY", frame.shape[0] / 2)
        fps_actual = getattr(self.config, "TARGET_FPS", 30.0)

        body_orientations = self.pose_estimator.get_orientations(frame, detections)

        for det in detections:
            track_id = det.get("track_id")
            z_meters = det.get("distance") or det.get("distance_m")

            # Nos aseguramos de tener un track_id válido y distancia
            if track_id is not None and z_meters is not None and z_meters > 0:
                current_frame_ids.add(track_id)
                x1, y1, x2, y2 = det["bbox"]
                
                feet_x = int((x1 + x2) / 2)
                feet_y = int(y2)
                x_meters = ((feet_x - cx) * z_meters) / fx

                # 1. GARANTIZAMOS QUE EL TRACKER EXISTA EN EL DICCIONARIO
                if track_id not in self.trackers:
                    self.trackers[track_id] = KalmanPredictor(fps=fps_actual)

                # 2. Actualizamos Kalman con las posiciones físicas
                self.trackers[track_id].update(x_meters, z_meters)

                # 3. Calculamos la trayectoria con Pose o con Kalman Fallback
                if track_id in body_orientations:
                    body_dx, body_dy = body_orientations[track_id]
                    det["future_path"] = self.trackers[track_id].predict_path_pixels_pose(
                        feet_x=feet_x,
                        feet_y=feet_y,
                        body_dx=body_dx,
                        body_dy=body_dy,
                        steps=8,
                        arrow_length=45
                    )
                else:
                    det["future_path"] = self.trackers[track_id].predict_path_pixels(
                        seconds_ahead=1.2,
                        steps=8,
                        fx=fx,
                        cx=cx,
                        feet_x=feet_x,
                        feet_y=feet_y,
                        cy=cy,
                        bbox=det["bbox"]
                    )
            
            print(f"DEBUG -> ID: {track_id}, tiene_path: {'future_path' in det}")

        # Limpieza de memoria para objetos que salen del cuadro
        lost_ids = set(self.trackers.keys()) - current_frame_ids
        for lost_id in lost_ids:
            del self.trackers[lost_id]


        # 6. Visualización (UN SOLO DRAW AL FINAL)
        start_visualization = time.perf_counter()
        output = self.visualizer.draw(frame, detections)
        visualization_time = (time.perf_counter() - start_visualization) * 1000

        # 7. Cálculo de métricas y renderizado de la UI
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
        
        self.trackers.clear()

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
