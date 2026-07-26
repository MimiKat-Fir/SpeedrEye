"""OpenCV visualization for SpeedrEye detections and runtime metrics."""

import cv2
import numpy as np


class Visualizer:
    # Resolución de referencia sobre la que están calibrados TEXT_SCALE,
    # BOX_THICKNESS, etc. en Config. El factor de escala compara el ancho
    # real del frame contra esta referencia, para que el overlay se vea
    # igual de legible tanto en un vídeo de 480p como en uno de 4K.
    REFERENCE_WIDTH = 1280
    MIN_SCALE, MAX_SCALE = 0.5, 3.0

    def __init__(self, config):
        self.config = config
        self._cached_shape = None
        self._set_scaled_defaults(self.REFERENCE_WIDTH)

    def _set_scaled_defaults(self, frame_width):
        scale = np.clip(frame_width / self.REFERENCE_WIDTH, self.MIN_SCALE, self.MAX_SCALE)
        self.scale = scale
        self.font_scale = self.config.TEXT_SCALE * scale
        self.text_thickness = max(1, round(self.config.TEXT_THICKNESS * scale))
        self.box_thickness = max(1, round(self.config.BOX_THICKNESS * scale))
        self.margin = max(4, round(10 * scale))
        self.line_height = max(14, round(24 * scale))
        self.arrow_size = round(10 * scale)
        self.point_radius = max(2, round(4 * scale))

    def _sync_scale(self, frame_shape):
        """Recalcula el factor de escala solo si cambia la resolución del frame."""
        w = frame_shape[1]
        if (w,) == self._cached_shape:
            return
        self._cached_shape = (w,)
        self._set_scaled_defaults(w)

    def _put_text_outlined(self, frame, text, org, color, font_scale=None, thickness=None):
        """Texto con contorno negro para que se lea igual sobre cualquier fondo."""
        font_scale = font_scale if font_scale is not None else self.font_scale
        thickness = thickness if thickness is not None else self.text_thickness
        cv2.putText(frame, text, org, self.config.FONT, font_scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, org, self.config.FONT, font_scale, color,
                    thickness, cv2.LINE_AA)

    def draw_trajectory(self, frame, detection, color):
        future_path = detection.get("future_path")

        if future_path is None or len(future_path) < 2:
            return

        x1, y1, x2, y2 = detection["bbox"]
        start_pt = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        end_pt = (int(future_path[-1][0]), int(future_path[-1][1]))

        cv2.line(frame, start_pt, end_pt, color, self.box_thickness + 1, cv2.LINE_AA)

        angle = np.arctan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])
        p1 = (
            int(end_pt[0] - self.arrow_size * np.cos(angle - np.pi / 6)),
            int(end_pt[1] - self.arrow_size * np.sin(angle - np.pi / 6))
        )
        p2 = (
            int(end_pt[0] - self.arrow_size * np.cos(angle + np.pi / 6)),
            int(end_pt[1] - self.arrow_size * np.sin(angle + np.pi / 6))
        )
        pts = np.array([end_pt, p1, p2], np.int32)
        cv2.fillPoly(frame, [pts], color, lineType=cv2.LINE_AA)
        cv2.circle(frame, start_pt, self.point_radius, color, -1, cv2.LINE_AA)

    def draw_detection(self, frame, detection):
        """
        Dibuja bbox + etiqueta + trayectoria. La alerta ya NO se marca por
        caja individual (era ruido visual con varias detecciones a la vez);
        la señal de alerta es el trapecio en rojo + el banner superior.
        """
        x1, y1, x2, y2 = detection["bbox"]
        class_id = detection["class"]
        confidence = detection["conf"]
        distance = detection.get("distance")

        color = self.config.CLASS_COLORS.get(class_id, (255, 255, 255))

        self.draw_trajectory(frame, detection, color)

        label = self.config.CLASS_NAMES.get(class_id, "Objeto")
        text = f"{label} {confidence:.2f}"
        if distance is not None:
            text += f" {distance:.1f}m"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness, cv2.LINE_AA)

        (text_width, text_height), _ = cv2.getTextSize(
            text, self.config.FONT, self.font_scale, self.text_thickness
        )

        pad = max(2, round(3 * self.scale))
        label_top = max(0, y1 - text_height - 2 * pad)
        cv2.rectangle(frame, (x1, label_top), (x1 + text_width + 2 * pad, y1), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (x1 + pad, max(text_height, y1 - pad)),
            self.config.FONT, self.font_scale, color, self.text_thickness, cv2.LINE_AA
        )

    def draw_alert_zone(self, frame, alert_zone):
        if alert_zone is None:
            return

        polygon = alert_zone.get("polygon")
        if polygon is None:
            return

        color = (0, 0, 255) if alert_zone.get("alert") else (255, 0, 0)
        cv2.polylines(frame, [polygon], True, color, self.box_thickness + 2, cv2.LINE_AA)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    def _draw_alert_banner(self, frame):
        """Aviso compacto arriba a la derecha, solo mientras haya alerta activa."""
        text = "COLISION INMINENTE"
        font_scale = self.font_scale * 1.5
        thickness = self.text_thickness + 1

        (tw, th), baseline = cv2.getTextSize(text, self.config.FONT, font_scale, thickness)
        pad = round(8 * self.scale)

        x2 = frame.shape[1] - self.margin
        x1 = x2 - tw - 2 * pad
        y1 = self.margin
        y2 = y1 + th + baseline + 2 * pad

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), max(1, round(self.scale)), cv2.LINE_AA)
        cv2.putText(
            frame, text, (x1 + pad, y2 - pad - baseline // 2),
            self.config.FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    def draw_ui(self, frame, metrics):
        """
        Solo las métricas que de verdad importan de un vistazo: FPS,
        detecciones activas y tiempo total por frame. Fijas a la
        izquierda, con contorno + panel semitransparente para que se
        lean igual sobre cualquier fondo del vídeo.
        """
        fps = metrics["fps"]
        # FPS coloreado por umbral: aviso visual instantáneo de bajón de rendimiento.
        if fps >= 25:
            fps_color = (0, 220, 0)
        elif fps >= 15:
            fps_color = (0, 200, 255)
        else:
            fps_color = (0, 0, 255)

        lines = (
            (f"FPS: {fps:.1f}", fps_color),
            (f"Detecciones: {metrics['detections']}", self.config.UI_TEXT_COLOR),
            (f"Frame: {metrics['total_time']:.0f}ms", self.config.UI_TEXT_COLOR),
        )

        panel_w = round(190 * self.scale)
        panel_h = round(12 * self.scale) + len(lines) * self.line_height

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        x = self.margin
        y = round(22 * self.scale)
        for text, color in lines:
            self._put_text_outlined(frame, text, (x, y), color)
            y += self.line_height

    def draw(self, frame, detections, alert_zone=None, metrics=None):
        self._sync_scale(frame.shape)
        output = frame.copy()

        # El trapecio se dibuja siempre que exista (venga de zona manual o
        # por defecto), haya o no detecciones en el frame actual.
        if alert_zone is not None:
            self.draw_alert_zone(output, alert_zone)

        for detection in detections:
            self.draw_detection(output, detection)

        if alert_zone is not None and alert_zone.get("alert"):
            self._draw_alert_banner(output)

        if metrics:
            self.draw_ui(output, metrics)

        return output