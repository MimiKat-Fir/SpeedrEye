import cv2
import numpy as np


class AlertSystem:
    """
    Sistema de alerta por invasión de la zona de seguridad.
    """

    def __init__(self, config):
        self.config = config

    def get_alert_zone(self, frame_shape):
        """
        Devuelve los cuatro vértices de la zona de seguridad.
        """

        h, w = frame_shape[:2]

        center_x = w // 2

        bottom_y = h
        top_y = h - self.config.ALERT_ZONE_HEIGHT

        bottom_half = self.config.ALERT_ZONE_BOTTOM_WIDTH // 2
        top_half = self.config.ALERT_ZONE_TOP_WIDTH // 2

        zone = np.array([
            [center_x - bottom_half, bottom_y],
            [center_x + bottom_half, bottom_y],
            [center_x + top_half, top_y],
            [center_x - top_half, top_y]
        ], dtype=np.int32)

        return zone

    def process(self, detections, frame_shape):
        """
        Comprueba si alguna trayectoria futura invade la zona.
        """

        zone = self.get_alert_zone(frame_shape)

        for det in detections:

            det["alert"] = False

            future = det.get("future_path")

            if future is None:
                continue

            for point in future:

                inside = cv2.pointPolygonTest(
                    zone,
                    (float(point[0]), float(point[1])),
                    False
                )

                if inside >= 0:
                    det["alert"] = True
                    break

        return detections, zone