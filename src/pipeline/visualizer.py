"""OpenCV visualization for SpeedrEye detections and runtime metrics."""

import cv2


class Visualizer:
    def __init__(self, config):
        self.config = config

    def draw_detection(self, frame, detection):
        x1, y1, x2, y2 = detection["bbox"]
        class_id = detection["class"]
        confidence = detection["conf"]
        color = self.config.CLASS_COLORS.get(class_id, (255, 255, 255))
        label = self.config.CLASS_NAMES.get(class_id, "Objeto")
        text = f"{label} {confidence:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            self.config.BOX_THICKNESS,
        )
        (text_width, text_height), _ = cv2.getTextSize(
            text,
            self.config.FONT,
            self.config.TEXT_SCALE,
            self.config.TEXT_THICKNESS,
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
            self.config.TEXT_THICKNESS,
        )

    def draw_ui(self, frame, metrics):
        lines = (
            f"FPS: {metrics['fps']:.1f}",
            f"Dets: {metrics['detections']}",
            f"YOLO: {metrics['detection_time']:.0f}ms",
            f"Total: {metrics['total_time']:.0f}ms",
        )
        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (frame.shape[1] - 150, 30 + index * 22),
                self.config.FONT,
                0.4,
                self.config.UI_TEXT_COLOR,
                1,
            )

    def draw(self, frame, detections, metrics=None):
        output = frame.copy()
        for detection in detections:
            self.draw_detection(output, detection)
        if metrics:
            self.draw_ui(output, metrics)
        return output
