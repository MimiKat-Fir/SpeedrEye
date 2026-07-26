#!/usr/bin/env python3
"""SpeedrEye detection pipeline."""

import argparse
import sys
import time
import json
from collections import deque
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.calibration import CameraCalibrator
from pipeline.config import Config
from pipeline.detector import Detector
from pipeline.distance import build_distance_estimator
from pipeline.visualizer import Visualizer
from pipeline.tracking import KalmanPredictor
from pipeline.tracking.pose_estimator import PoseOrientationEstimator
from pipeline.alert.system import AlertSystem


class SpeedrEyePipeline:
    def __init__(self, config, video_path=None, distance_method=None):
        self.config = config
        self.fps_buffer = deque(maxlen=config.FPS_BUFFER_SIZE)
        self.frame_count = 0
        self.trackers = {}

        # Métricas de rendimiento
        self.performance_metrics = {
            'timings': [],
            'total_frames': 0,
            'total_time': 0,
        }

        # Calibración
        self.calibrator = CameraCalibrator(config)
        if video_path and Path(video_path).exists():
            self.calibrator.calibrate_from_video(video_path, num_frames=20)

        self.detector = Detector(config)
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
        self.alert_system = AlertSystem(config)
        self.pose_estimator = PoseOrientationEstimator()

        # Contador para saltar frames de pose
        self.pose_counter = 0

        # Variables para ajuste manual de la zona de alerta
        self.alert_zone_polygon = None
        self.alert_zone_configured = False

    def configure_alert_zone(self, frame):
        """
        Permite ajustar manualmente la zona de alerta.
        Versión mejorada con restricciones inteligentes.
        """
        print("\n🔧 Configuración manual de la zona de alerta")
        print("   Arrastra los círculos de las esquinas para ajustar el trapecio")
        print("   Presiona ENTER cuando esté listo")
        print("   Presiona ESC para cancelar\n")

        h, w = frame.shape[:2]

        # Valores iniciales: las esquinas inferiores arrancan en las
        # esquinas inferiores reales del frame (0 y w), no centradas con
        # un ancho fijo — así el punto de partida ya cubre todo el carril
        # visible más cercano a la cámara.
        top_width = min(150, w // 4)
        height = min(200, h // 3)

        # Puntos del trapecio (orden: superior-izquierdo, superior-derecho, inferior-derecho, inferior-izquierdo)
        cx = w // 2
        bottom_y = h
        top_y = h - height

        points = np.array([
            [cx - top_width // 2, top_y],
            [cx + top_width // 2, top_y],
            [w, bottom_y],
            [0, bottom_y]
        ], dtype=np.float32)

        # Variables de arrastre
        dragging = -1
        drag_radius = 25
        last_mouse_pos = None

        def draw_zone(img):
            viz = img.copy()
            pts = points.astype(np.int32)

            # Relleno semitransparente
            overlay = viz.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, viz, 0.85, 0, viz)

            # Líneas del trapecio
            cv2.polylines(viz, [pts], True, (0, 255, 0), 2)

            # Cuadrícula de referencia
            for i in range(1, 6):
                alpha = i / 6
                top_pt = pts[0] + alpha * (pts[1] - pts[0])
                bottom_pt = pts[3] + alpha * (pts[2] - pts[3])
                cv2.line(viz, tuple(top_pt.astype(np.int32)),
                        tuple(bottom_pt.astype(np.int32)),
                        (0, 255, 0), 1, cv2.LINE_AA)

            # Círculos en las esquinas (más grandes y visibles)
            for i, pt in enumerate(points):
                cv2.circle(viz, tuple(pt.astype(np.int32)), drag_radius, (0, 0, 255), 2)
                cv2.circle(viz, tuple(pt.astype(np.int32)), 8, (0, 0, 255), -1)
                cv2.putText(viz, f"{i+1}",
                           (int(pt[0]) - 10, int(pt[1]) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(viz, "Ajusta la zona de alerta (arrastra los circulos rojos)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(viz, "ENTER: aceptar   ESC: cancelar   R: resetear",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return viz

        def constrain_point(idx, new_pos):
            """Aplica restricciones inteligentes para mantener el trapecio"""
            x, y = new_pos

            if idx == 0:  # Superior-izquierdo
                top_right = points[1]
                bottom_left = points[3]
                x = min(x, top_right[0] - 10)
                y = min(y, bottom_left[1] - 10)
            elif idx == 1:  # Superior-derecho
                top_left = points[0]
                bottom_right = points[2]
                x = max(x, top_left[0] + 10)
                y = min(y, bottom_right[1] - 10)
            elif idx == 2:  # Inferior-derecho
                top_right = points[1]
                bottom_left = points[3]
                x = max(x, bottom_left[0] + 10)
                y = max(y, top_right[1] + 10)
            elif idx == 3:  # Inferior-izquierdo
                top_left = points[0]
                bottom_right = points[2]
                x = min(x, bottom_right[0] - 10)
                y = max(y, top_left[1] + 10)

            x = max(0, min(w, x))
            y = max(0, min(h, y))

            return np.array([x, y], dtype=np.float32)

        def reset_zone():
            """Restablece la zona a los valores por defecto"""
            nonlocal points
            top_width = min(150, w // 4)
            height = min(200, h // 3)
            cx = w // 2
            bottom_y = h
            top_y = h - height
            points = np.array([
                [cx - top_width // 2, top_y],
                [cx + top_width // 2, top_y],
                [w, bottom_y],
                [0, bottom_y]
            ], dtype=np.float32)

        def mouse_callback(event, x, y, flags, param):
            nonlocal dragging, points, last_mouse_pos

            if event == cv2.EVENT_LBUTTONDOWN:
                for i, pt in enumerate(points):
                    if np.linalg.norm(pt - np.array([x, y])) < drag_radius:
                        dragging = i
                        last_mouse_pos = np.array([x, y], dtype=np.float32)
                        break

            elif event == cv2.EVENT_MOUSEMOVE:
                if dragging != -1:
                    current_pos = np.array([x, y], dtype=np.float32)
                    delta = current_pos - last_mouse_pos if last_mouse_pos is not None else np.array([0, 0])

                    new_pos = points[dragging] + delta
                    constrained = constrain_point(dragging, new_pos)

                    points[dragging] = constrained
                    last_mouse_pos = current_pos.copy()

                    viz = draw_zone(frame)
                    cv2.imshow("Ajuste de Zona de Alerta", viz)

            elif event == cv2.EVENT_LBUTTONUP:
                dragging = -1
                last_mouse_pos = None

        # Ventana redimensionable, tamaño inicial acorde a la resolución real
        # (funciona igual en Linux y Windows).
        cv2.namedWindow("Ajuste de Zona de Alerta", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("Ajuste de Zona de Alerta", min(w, 1280), min(h, 720))
        cv2.setMouseCallback("Ajuste de Zona de Alerta", mouse_callback)

        while True:
            viz = draw_zone(frame)
            cv2.imshow("Ajuste de Zona de Alerta", viz)

            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # ENTER
                self.alert_zone_polygon = points.astype(np.int32)
                self.alert_zone_configured = True
                break
            elif key == 27:  # ESC
                self.alert_zone_configured = False
                break
            elif key == ord('r') or key == ord('R'):  # Reset
                reset_zone()
                print("🔄 Zona restablecida a valores por defecto")

        cv2.destroyWindow("Ajuste de Zona de Alerta")

        if self.alert_zone_configured:
            print("✅ Zona de alerta configurada manualmente")
        else:
            print("⏭️ Usando zona de alerta por defecto")

    def process_frame(self, frame):
        start_total = time.perf_counter()

        # 1. Detección
        start_detection = time.perf_counter()
        detections = self.detector.detect(frame)
        detection_time = (time.perf_counter() - start_detection) * 1000

        # 2. Recalibración (opcional, desactivada por defecto)
        start_calib = time.perf_counter()
        if (self.config.ENABLE_SCENE_RECALIBRATION
                and self.frame_count % self.config.SCENE_RECALIBRATION_INTERVAL == 0
                and self.calibrator.is_calibrated):
            self.calibrator.update_scene_params(frame)
        calib_time = (time.perf_counter() - start_calib) * 1000

        # 3. Filtrado por horizonte
        if self.calibrator.horizon_line is not None:
            horizon_y = self.calibrator.horizon_line
            detections = [
                d for d in detections
                if (d["bbox"][1] + d["bbox"][3]) / 2 > horizon_y
            ]

        # 4. Distancia
        start_distance = time.perf_counter()
        if detections and self.distance_estimator is not None:
            detections = self.distance_estimator.estimate(detections, frame.shape)
        distance_time = (time.perf_counter() - start_distance) * 1000

        # 5. Predicción (Kalman + Pose fusionados) - Optimizado
        start_prediction = time.perf_counter()

        current_frame_ids = set()

        if detections:
            fx = getattr(self.config, "FOCAL_LENGTH", 800.0)

            self.pose_counter += 1
            if self.pose_counter >= self.config.POSE_FRAME_SKIP:
                body_orientations = self.pose_estimator.get_orientations(frame, detections)
                self.pose_counter = 0
            else:
                body_orientations = {}

            for det in detections:
                track_id = det.get("track_id")
                z_meters = det.get("distance") or det.get("distance_m")

                if track_id is not None and z_meters is not None and z_meters > 0:
                    current_frame_ids.add(track_id)
                    x1, y1, x2, y2 = det["bbox"]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    x_meters = ((center_x - getattr(self.config, "CX", frame.shape[1] / 2)) * z_meters) / fx

                    if track_id not in self.trackers:
                        self.trackers[track_id] = KalmanPredictor(
                            fps=getattr(self.config, "TARGET_FPS", 30.0), history_size=6
                        )

                    self.trackers[track_id].update(x_meters, z_meters)

                    det["future_path"] = self.trackers[track_id].predict_path_pixels(
                        fx=fx,
                        center_x=center_x,
                        center_y=center_y,
                        steps=self.config.PREDICTION_STEPS,
                        body_orientation=body_orientations.get(track_id),
                        seconds_ahead=self.config.PREDICTION_SECONDS,
                    )

        # Limpieza de trackers perdidos (incluye el caso de frame sin detecciones)
        lost_ids = set(self.trackers.keys()) - current_frame_ids
        for lost_id in lost_ids:
            del self.trackers[lost_id]

        prediction_time = (time.perf_counter() - start_prediction) * 1000

        # 6. Alertas — el trapecio se calcula y se dibuja siempre (haya o
        # no detecciones); lo único que se salta sin detecciones es el
        # bucle de evaluación por objeto, que con lista vacía no hace nada.
        start_alert = time.perf_counter()

        manual_polygon = (
            self.alert_zone_polygon
            if (self.alert_zone_configured and self.alert_zone_polygon is not None)
            else None
        )
        detections, alert_zone = self.alert_system.process(detections, frame.shape, polygon=manual_polygon)

        alert_time = (time.perf_counter() - start_alert) * 1000

        # 7. Visualización
        start_visualization = time.perf_counter()
        output = self.visualizer.draw(frame, detections, alert_zone=alert_zone)
        visualization_time = (time.perf_counter() - start_visualization) * 1000

        # 8. Métricas
        total_time = (time.perf_counter() - start_total) * 1000
        self.fps_buffer.append(1000 / total_time if total_time > 0 else 0)
        self.frame_count += 1

        self.performance_metrics['timings'].append({
            'frame': self.frame_count,
            'detection_ms': detection_time,
            'distance_ms': distance_time,
            'prediction_ms': prediction_time,
            'alert_ms': alert_time,
            'visualization_ms': visualization_time,
            'total_ms': total_time,
            'calib_time': calib_time
        })
        self.performance_metrics['total_frames'] = self.frame_count
        self.performance_metrics['total_time'] += total_time

        metrics = {
            "fps": np.mean(self.fps_buffer),
            "detections": len(detections),
            "detection_time": detection_time,
            "distance_time": distance_time,
            "prediction_time": prediction_time,
            "alert_time": alert_time,
            "visualization_time": visualization_time,
            "total_time": total_time,
            "distance_method": self.distance_method,
            "calibration_ms": calib_time
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

        # Leer primer frame: se usa para configurar la zona de alerta (si
        # aplica) y para dimensionar la ventana según la resolución real.
        ret, first_frame = cap.read()
        if ret:
            if self.config.ENABLE_ALERT_SYSTEM:
                self.configure_alert_zone(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # siempre rebobinar, se use o no el frame

        # Ventana redimensionable de verdad en Linux y Windows: por defecto
        # cv2.imshow usa WINDOW_AUTOSIZE, que NO permite estirar la ventana.
        # WINDOW_NORMAL lo permite; WINDOW_KEEPRATIO evita que la imagen se
        # deforme al redimensionar. El tamaño inicial se ajusta a la
        # resolución real del vídeo (con tope, para no abrir gigante en 4K
        # ni minúsculo en baja resolución).
        window_name = "SpeedrEye"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        if ret:
            h, w = first_frame.shape[:2]
            cv2.resizeWindow(window_name, min(w, 1280), min(h, 720))

        print("\n🚀 Iniciando procesamiento en tiempo real...\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            output, _, _ = self.process_frame(frame)
            cv2.imshow(window_name, output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Guardar métricas de rendimiento
        self.save_performance_metrics()

    def save_performance_metrics(self):
        """Guarda las métricas de rendimiento en un archivo JSON"""
        if not self.performance_metrics['timings']:
            return

        timings = self.performance_metrics['timings']
        avg_timings = {
            'avg_detection_ms': np.mean([t['detection_ms'] for t in timings]),
            'avg_distance_ms': np.mean([t['distance_ms'] for t in timings]),
            'avg_prediction_ms': np.mean([t['prediction_ms'] for t in timings]),
            'avg_alert_ms': np.mean([t['alert_ms'] for t in timings]),
            'avg_visualization_ms': np.mean([t['visualization_ms'] for t in timings]),
            'avg_total_ms': np.mean([t['total_ms'] for t in timings]),
            'max_total_ms': np.max([t['total_ms'] for t in timings]),
            'min_total_ms': np.min([t['total_ms'] for t in timings]),
            'total_frames': len(timings),
            'total_time_sec': self.performance_metrics['total_time'] / 1000,
            'avg_fps': np.mean(self.fps_buffer) if self.fps_buffer else 0,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.config.RESULTS_DIR / "performance" / f"performance_{timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'summary': avg_timings,
                'frames': timings,
                'config': {
                    'distance_method': self.distance_method,
                    'pose_frame_skip': self.config.POSE_FRAME_SKIP,
                    'prediction_steps': self.config.PREDICTION_STEPS,
                    'prediction_seconds': self.config.PREDICTION_SECONDS,
                }
            }, f, indent=2)

        print(f"\n📊 Métricas de rendimiento guardadas en: {output_path}")
        print(f"   FPS promedio: {avg_timings['avg_fps']:.1f}")
        print(f"   Tiempo promedio por frame: {avg_timings['avg_total_ms']:.1f}ms")
        print(f"   Detección: {avg_timings['avg_detection_ms']:.1f}ms")
        print(f"   Distancia: {avg_timings['avg_distance_ms']:.1f}ms")
        print(f"   Predicción: {avg_timings['avg_prediction_ms']:.1f}ms")
        print(f"   Alertas: {avg_timings['avg_alert_ms']:.1f}ms")
        print(f"   Visualización: {avg_timings['avg_visualization_ms']:.1f}ms")


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