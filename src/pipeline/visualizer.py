"""OpenCV visualization for SpeedrEye detections and runtime metrics."""

import cv2
import numpy as np


class Visualizer:
    def __init__(self, config):
        self.config = config

    def draw_trajectory(self, frame, detection, color):
        future_path = detection.get("future_path")

        if future_path is None or len(future_path) < 2:
            return

        x1, y1, x2, y2 = detection["bbox"]

        # 🔥 PUNTO INICIAL: CENTRO DEL BBOX
        start_pt = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

        end_pt = (
            int(future_path[-1][0]),
            int(future_path[-1][1])
        )

        # Línea de trayectoria
        cv2.line(
            frame,
            start_pt,
            end_pt,
            color,
            2,
            cv2.LINE_AA
        )

        # Flecha final
        angle = np.arctan2(
            end_pt[1] - start_pt[1],
            end_pt[0] - start_pt[0]
        )

        arrow_size = 10

        p1 = (
            int(end_pt[0] - arrow_size * np.cos(angle - np.pi / 6)),
            int(end_pt[1] - arrow_size * np.sin(angle - np.pi / 6))
        )

        p2 = (
            int(end_pt[0] - arrow_size * np.cos(angle + np.pi / 6)),
            int(end_pt[1] - arrow_size * np.sin(angle + np.pi / 6))
        )

        pts = np.array([end_pt, p1, p2], np.int32)
        cv2.fillPoly(frame, [pts], color)

        # Círculo en el centro
        cv2.circle(frame, start_pt, 4, color, -1)

    def draw_detection(self, frame, detection):
        x1, y1, x2, y2 = detection["bbox"]
        class_id = detection["class"]
        confidence = detection["conf"]
        distance = detection.get("distance")

        color = self.config.CLASS_COLORS.get(class_id, (255, 255, 255))

        # Trayectoria futura
        self.draw_trajectory(frame, detection, color)

        label = self.config.CLASS_NAMES.get(class_id, "Objeto")
        text = f"{label} {confidence:.2f}"
        if distance is not None:
            text += f" {distance:.1f}m"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.BOX_THICKNESS)

        (text_width, text_height), _ = cv2.getTextSize(
            text,
            self.config.FONT,
            self.config.TEXT_SCALE,
            self.config.TEXT_THICKNESS
        )

        label_top = max(0, y1 - text_height - 6)
        cv2.rectangle(frame, (x1, label_top), (x1 + text_width + 6, y1), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (x1 + 3, max(text_height, y1 - 3)),
            self.config.FONT,
            self.config.TEXT_SCALE,
            color,
            self.config.TEXT_THICKNESS
        )

        # Alerta individual
        if detection.get("alert"):
            cv2.putText(
                frame,
                "⚠️ PELIGRO",
                (x1, y2 + 30),
                self.config.FONT,
                0.8,
                (0, 0, 255),
                2
            )

    def draw_alert_zone(self, frame, alert_zone):
        if alert_zone is None:
            return

        polygon = alert_zone.get("polygon")
        if polygon is None:
            return

        color = (0, 0, 255) if alert_zone.get("alert") else (255, 0, 0)
        
        # Dibujar trapecio
        cv2.polylines(frame, [polygon], True, color, 3)
        
        # Relleno semitransparente
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    def draw_ui(self, frame, metrics):
        lines = (
            f"FPS: {metrics['fps']:.1f}",
            f"Dets: {metrics['detections']}",
            f"YOLO: {metrics['detection_time']:.0f}ms",
            f"Dist: {metrics['distance_time']:.1f}ms",
            f"Pred: {metrics['prediction_time']:.1f}ms",
            f"Alert: {metrics['alert_time']:.1f}ms",
            f"Draw: {metrics['visualization_time']:.1f}ms",
            f"Total: {metrics['total_time']:.0f}ms",
        )

        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (frame.shape[1] - 170, 30 + index * 22),
                self.config.FONT,
                0.4,
                self.config.UI_TEXT_COLOR,
                1
            )

    def draw(self, frame, detections, alert_zone=None, metrics=None):
        output = frame.copy()

        # Zona de seguridad
        if alert_zone is not None:
            self.draw_alert_zone(output, alert_zone)

        # Detecciones
        for detection in detections:
            self.draw_detection(output, detection)

        # Colisión global
        if any(detection.get("alert") for detection in detections):
            cv2.putText(
                output,
                "⚠️ COLISION INMINENTE!",
                (50, 80),
                self.config.FONT,
                1.5,
                (0, 0, 255),
                3
            )

        if metrics:
            self.draw_ui(output, metrics)

        return output