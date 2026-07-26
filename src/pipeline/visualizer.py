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

        # punto inicial: pies del objeto
        start_pt = (
            int((x1 + x2) / 2),
            int(y2)
        )

        # punto futuro
        end_pt = (
            int(future_path[-1][0]),
            int(future_path[-1][1])
        )


        # línea de trayectoria
        cv2.line(
            frame,
            start_pt,
            end_pt,
            color,
            3,
            cv2.LINE_AA
        )


        # flecha final
        angle = np.arctan2(
            end_pt[1] - start_pt[1],
            end_pt[0] - start_pt[0]
        )

        arrow_size = 12


        p1 = (
            int(end_pt[0] - arrow_size * np.cos(angle - np.pi / 6)),
            int(end_pt[1] - arrow_size * np.sin(angle - np.pi / 6))
        )

        p2 = (
            int(end_pt[0] - arrow_size * np.cos(angle + np.pi / 6)),
            int(end_pt[1] - arrow_size * np.sin(angle + np.pi / 6))
        )


        pts = np.array(
            [
                end_pt,
                p1,
                p2
            ],
            np.int32
        )

        cv2.fillPoly(
            frame,
            [pts],
            color
        )


        cv2.circle(
            frame,
            start_pt,
            4,
            color,
            -1
        )



    def draw_detection(self, frame, detection):

        x1, y1, x2, y2 = detection["bbox"]

        class_id = detection["class"]
        confidence = detection["conf"]

        distance = detection.get("distance")


        color = self.config.CLASS_COLORS.get(
            class_id,
            (255,255,255)
        )


        # trayectoria futura
        self.draw_trajectory(
            frame,
            detection,
            color
        )


        label = self.config.CLASS_NAMES.get(
            class_id,
            "Objeto"
        )


        text = f"{label} {confidence:.2f}"


        if distance is not None:
            text += f" {distance:.1f}m"



        # caja detección

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            color,
            self.config.BOX_THICKNESS
        )



        (text_width,text_height),_ = cv2.getTextSize(
            text,
            self.config.FONT,
            self.config.TEXT_SCALE,
            self.config.TEXT_THICKNESS
        )


        label_top = max(
            0,
            y1-text_height-6
        )


        cv2.rectangle(
            frame,
            (x1,label_top),
            (x1+text_width+6,y1),
            (0,0,0),
            -1
        )


        cv2.putText(
            frame,
            text,
            (
                x1+3,
                max(text_height,y1-3)
            ),
            self.config.FONT,
            self.config.TEXT_SCALE,
            color,
            self.config.TEXT_THICKNESS
        )



        # alerta individual

        if detection.get("alert"):

            cv2.putText(
                frame,
                "PELIGRO",
                (
                    x1,
                    y2+30
                ),
                self.config.FONT,
                0.8,
                (0,0,255),
                2
            )



        # posición futura si existe

        future_pos = detection.get("future_pos_3s")

        if future_pos:

            x_fut,z_fut = future_pos

            text_fut = (
                f"En 3s: "
                f"({x_fut:+.1f}m,{z_fut:.1f}m)"
            )


            cv2.putText(
                frame,
                text_fut,
                (
                    x1,
                    y2+15
                ),
                self.config.FONT,
                self.config.TEXT_SCALE*0.8,
                color,
                1
            )



    def draw_alert_zone(self, frame, alert_zone):

        if alert_zone is None:
            return


        polygon = alert_zone.get("polygon")

        if polygon is None:
            return



        if alert_zone.get("alert"):

            color = (0,0,255)

        else:

            color = (255,0,0)



        cv2.polylines(
            frame,
            [
                polygon
            ],
            True,
            color,
            3
        )



    def draw_ui(self, frame, metrics):

        lines = (
            f"FPS: {metrics['fps']:.1f}",
            f"Dets: {metrics['detections']}",
            f"YOLO: {metrics['detection_time']:.0f}ms",
            f"Distance: {metrics['distance_time']:.1f}ms",
            f"Draw: {metrics['visualization_time']:.1f}ms",
            f"Total: {metrics['total_time']:.0f}ms",
        )


        for index,text in enumerate(lines):

            cv2.putText(
                frame,
                text,
                (
                    frame.shape[1]-150,
                    30+index*22
                ),
                self.config.FONT,
                0.4,
                self.config.UI_TEXT_COLOR,
                1
            )



    def draw(
        self,
        frame,
        detections,
        alert_zone=None,
        metrics=None
    ):


        output = frame.copy()


        # zona seguridad coche

        if alert_zone is not None:

            self.draw_alert_zone(
                output,
                alert_zone
            )



        collision = False


        for detection in detections:


            if detection.get("alert"):

                collision = True



            self.draw_detection(
                output,
                detection
            )



        if collision:


            cv2.putText(
                output,
                "COLISION!",
                (
                    50,
                    80
                ),
                self.config.FONT,
                2,
                (0,0,255),
                4
            )



        if metrics:

            self.draw_ui(
                output,
                metrics
            )


        return output