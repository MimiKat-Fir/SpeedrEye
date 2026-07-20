"""Direct per-object distance regression from shared YOLO features."""

import torch

from .head import BackboneFeatureCapture, load_distance_head, prepare_distance_inputs


class DirectDistanceEstimator:
    def __init__(self, detector, config):
        self.config = config
        self.capture = BackboneFeatureCapture(detector.model.model)
        self.head = load_distance_head(
            config.DIRECT_DISTANCE_WEIGHTS,
            "direct",
            config.YOLO_MODEL_PATH,
        )

    def estimate(self, detections, original_shape):
        if not detections:
            return detections
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
            distances = self.head(feature_map, rois, box_features).exp()
            distances = distances.clamp(self.config.MIN_DISTANCE, self.config.MAX_DISTANCE)

        for detection, distance in zip(detections, distances.tolist()):
            detection["distance"] = distance
            detection["distance_method"] = "direct"
        return detections
