"""Small regression head that reuses feature maps from the YOLO detector."""

from pathlib import Path

import torch
from torch import nn
from torchvision.ops import roi_align


class BackboneFeatureCapture:
    """Capture the feature pyramid passed to YOLO's detection head."""

    def __init__(self, detection_model):
        self.features = None
        detect_head = detection_model.model[-1]
        self.strides = tuple(float(value) for value in detect_head.stride)
        self._handle = detect_head.register_forward_pre_hook(self._capture)

    def _capture(self, _module, inputs):
        self.features = tuple(inputs[0])

    def close(self):
        self._handle.remove()


class DistanceRegressionHead(nn.Module):
    """Predict one log-distance value from an object ROI and its box geometry."""

    def __init__(self, feature_channels, hidden_channels=64, roi_size=3):
        super().__init__()
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.roi_size = roi_size
        self.regressor = nn.Sequential(
            nn.Linear(feature_channels + 4, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, feature_map, rois, box_features):
        pooled = roi_align(
            feature_map,
            rois,
            output_size=self.roi_size,
            spatial_scale=1.0,
            aligned=True,
        ).mean(dim=(-1, -2))
        return self.regressor(torch.cat((pooled, box_features), dim=1)).squeeze(1)


def prepare_distance_inputs(detections, original_shape, feature_map, stride):
    """Map original-image boxes to one letterboxed YOLO feature map."""

    original_height, original_width = original_shape[:2]
    feature_height, feature_width = feature_map.shape[-2:]
    input_height = feature_height * stride
    input_width = feature_width * stride

    gain = min(input_width / original_width, input_height / original_height)
    pad_x = (input_width - original_width * gain) / 2
    pad_y = (input_height - original_height * gain) / 2

    device = feature_map.device
    boxes = torch.tensor(
        [detection["bbox"] for detection in detections],
        dtype=feature_map.dtype,
        device=device,
    )
    network_boxes = boxes.clone()
    network_boxes[:, (0, 2)] = network_boxes[:, (0, 2)] * gain + pad_x
    network_boxes[:, (1, 3)] = network_boxes[:, (1, 3)] * gain + pad_y
    network_boxes[:, (0, 2)] *= feature_width / input_width
    network_boxes[:, (1, 3)] *= feature_height / input_height
    network_boxes[:, (0, 2)].clamp_(0, feature_width - 1)
    network_boxes[:, (1, 3)].clamp_(0, feature_height - 1)

    batch_indices = torch.zeros((len(detections), 1), dtype=boxes.dtype, device=device)
    rois = torch.cat((batch_indices, network_boxes), dim=1)

    box_width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1)
    box_height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1)
    box_features = torch.stack(
        (
            (boxes[:, 0] + boxes[:, 2]) / (2 * original_width),
            (boxes[:, 1] + boxes[:, 3]) / (2 * original_height),
            box_width / original_width,
            box_height / original_height,
        ),
        dim=1,
    )
    return rois, box_features


def load_distance_head(checkpoint_path, expected_mode):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing distance weights: {path}. Train them with notebooks/train_direct_distance.ipynb."
        )

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    mode = checkpoint.get("target_mode")
    if mode != expected_mode:
        raise ValueError(f"Expected '{expected_mode}' weights, found '{mode}' in {path}")

    head = DistanceRegressionHead(
        feature_channels=checkpoint["feature_channels"],
        hidden_channels=checkpoint.get("hidden_channels", 64),
        roi_size=checkpoint.get("roi_size", 3),
    )
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    return head
