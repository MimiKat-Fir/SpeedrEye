"""Camera geometry with an optional learned correction from YOLO features."""

from pathlib import Path

import torch

from .head import BackboneFeatureCapture, load_distance_head, prepare_distance_inputs


class GeometryGuidedDistanceEstimator:
    def __init__(self, detector, config):
        self.config = config
        self.capture = None
        self.head = None

        if Path(config.GEOMETRY_DISTANCE_WEIGHTS).exists():
            self.capture = BackboneFeatureCapture(detector.model.model)
            self.head = load_distance_head(
                config.GEOMETRY_DISTANCE_WEIGHTS,
                "geometry_guided",
                config.YOLO_MODEL_PATH,
            )

    def _geometry_distance(self, detection):
        box_height = max(detection["bbox"][3] - detection["bbox"][1], 1)
        object_height = self.config.CLASS_HEIGHTS[detection["class"]]
        distance = object_height * self.config.FOCAL_LENGTH / box_height
        return min(max(distance, self.config.MIN_DISTANCE), self.config.MAX_DISTANCE)

    def estimate(self, detections, original_shape):
        if not detections:
            return detections

        geometry = [self._geometry_distance(detection) for detection in detections]
        correction = [1.0] * len(detections)

        if self.head is not None:
            if not self.capture.features:
                raise RuntimeError("YOLO features were not captured before distance estimation")
            feature_map = self.capture.features[0]
            if feature_map.shape[1] != self.head.feature_channels:
                raise RuntimeError(
                    f"Distance head expects {self.head.feature_channels} feature channels, "
                    f"but YOLO produced {feature_map.shape[1]}"
                )
            self.head.to(feature_map.device)
            rois, box_features = prepare_distance_inputs(
                detections,
                original_shape,
                feature_map,
                self.capture.strides[0],
            )
            with torch.inference_mode():
                correction = self.head(feature_map, rois, box_features).exp().tolist()

        method = "geometry_guided" if self.head is not None else "geometry"
        for detection, base_distance, scale in zip(detections, geometry, correction):
            distance = base_distance * scale
            detection["distance"] = min(
                max(distance, self.config.MIN_DISTANCE),
                self.config.MAX_DISTANCE,
            )
            detection["distance_method"] = method
        return detections
