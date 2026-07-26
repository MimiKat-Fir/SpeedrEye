"""Continue direct-distance training locally on a Linux CUDA system."""

from pathlib import Path
import hashlib
import json
import random
import urllib.request
import zipfile

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from src.pipeline.distance.head import (
    BackboneFeatureCapture,
    DistanceRegressionHead,
    prepare_distance_inputs,
)


# Local training configuration
PROJECT_ROOT = Path(__file__).resolve().parent
KITTI_ROOT = PROJECT_ROOT / "data" / "kitti_raw"
TRAINING_DIR = KITTI_ROOT / "training"
DETECTOR_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "speedreye_best.pt"
BEST_WEIGHTS = PROJECT_ROOT / "models" / "distance" / "direct_distance.pt"
LAST_WEIGHTS = PROJECT_ROOT / "models" / "distance" / "direct_distance_last.pt"
METRICS_PATH = PROJECT_ROOT / "models" / "distance" / "direct_distance_metrics.json"
TENSORBOARD_ROOT = PROJECT_ROOT / "tensorboard" / "direct_distance"
RUN_NAME = "direct_distance_local_100"
TENSORBOARD_DIR = TENSORBOARD_ROOT / RUN_NAME

TOTAL_EPOCHS = 100
IMAGE_SIZE = 640
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
SCHEDULER_PATIENCE = 4
EARLY_STOPPING_PATIENCE = 12
MIN_IMPROVEMENT_M = 0.02
SEED = 42
MAX_DISTANCE_M = 60.0
TARGET_CLASSES = {"Pedestrian": 0, "Cyclist": 1}
DOWNLOAD_KITTI_IF_MISSING = True

CORE_ARCHIVES = {
    "data_object_image_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "data_object_label_2.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    "data_object_calib.zip": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
}


def load_torch_checkpoint(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_kitti_if_needed():
    image_dir = TRAINING_DIR / "image_2"
    label_dir = TRAINING_DIR / "label_2"
    if image_dir.exists() and label_dir.exists():
        return
    if not DOWNLOAD_KITTI_IF_MISSING:
        raise FileNotFoundError(f"KITTI is missing under {TRAINING_DIR}")

    KITTI_ROOT.mkdir(parents=True, exist_ok=True)
    for filename, url in CORE_ARCHIVES.items():
        archive = KITTI_ROOT / filename
        if not archive.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, archive)
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(archive) as compressed:
            compressed.extractall(KITTI_ROOT)


def parse_kitti_objects(path):
    objects = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields[0] not in TARGET_CLASSES:
            continue
        distance = float(fields[13])
        if 0.5 <= distance <= MAX_DISTANCE_M:
            objects.append(
                {
                    "class_name": fields[0],
                    "class_id": TARGET_CLASSES[fields[0]],
                    "bbox": tuple(float(value) for value in fields[4:8]),
                    "distance_m": distance,
                    "distance_source": "kitti_3d",
                    "truncated": float(fields[1]),
                    "occluded": int(fields[2]),
                }
            )
    return objects


def build_records():
    image_dir = TRAINING_DIR / "image_2"
    label_dir = TRAINING_DIR / "label_2"
    rows = []

    for label_path in tqdm(sorted(label_dir.glob("*.txt")), desc="Building targets"):
        frame_id = label_path.stem
        image_path = image_dir / f"{frame_id}.png"
        for obj in parse_kitti_objects(label_path):
            rows.append(
                {
                    **obj,
                    "frame_id": frame_id,
                    "image_path": str(image_path),
                }
            )

    records = pd.DataFrame(rows)
    if records.empty:
        raise RuntimeError("No Pedestrian/Cyclist targets were found in KITTI")
    return records


def split_frames(records):
    frame_ids = records["frame_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(SEED)
    rng.shuffle(frame_ids)
    split_index = int(0.8 * len(frame_ids))
    return frame_ids[:split_index].tolist(), frame_ids[split_index:].tolist()


def tensorboard_training_state():
    """Read epoch metadata missing from the legacy Colab checkpoint."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    latest = 0
    best_epoch = 0
    best_mae = float("inf")
    for event_path in TENSORBOARD_ROOT.rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        if "train/mae_m" in accumulator.Tags().get("scalars", []):
            events = accumulator.Scalars("train/mae_m")
            if events:
                latest = max(latest, max(event.step for event in events))
        if "validation/mae_m" in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars("validation/mae_m"):
                if event.value < best_mae:
                    best_mae = event.value
                    best_epoch = event.step
    return latest, best_epoch


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this local training script. "
        "Install a CUDA-enabled PyTorch build and verify torch.cuda.is_available()."
    )

DEVICE = torch.device("cuda")
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

download_kitti_if_needed()
if not DETECTOR_WEIGHTS.exists():
    raise FileNotFoundError(f"Missing detector weights: {DETECTOR_WEIGHTS}")
if not BEST_WEIGHTS.exists() and not LAST_WEIGHTS.exists():
    raise FileNotFoundError(
        f"No distance checkpoint found. Expected {BEST_WEIGHTS} from the Colab run."
    )

records = build_records()
train_frames, val_frames = split_frames(records)
records_by_frame = {
    frame_id: frame_rows.copy()
    for frame_id, frame_rows in records.groupby("frame_id", sort=False)
}

print("Device:", torch.cuda.get_device_name(0))
print("Training frames/objects:", len(train_frames), len(records[records["frame_id"].isin(train_frames)]))
print("Validation frames/objects:", len(val_frames), len(records[records["frame_id"].isin(val_frames)]))

detector = YOLO(str(DETECTOR_WEIGHTS)).model.to(DEVICE).eval()
for parameter in detector.parameters():
    parameter.requires_grad = False
capture = BackboneFeatureCapture(detector)
letterbox = LetterBox(new_shape=(IMAGE_SIZE, IMAGE_SIZE), auto=False, stride=32)


def load_frame_features(frame_id):
    frame_rows = records_by_frame[frame_id]
    image = cv2.imread(frame_rows.iloc[0]["image_path"])
    if image is None:
        raise FileNotFoundError(frame_rows.iloc[0]["image_path"])

    resized = letterbox(image=image)
    network_input = resized[:, :, ::-1].transpose(2, 0, 1).copy()
    network_input = torch.from_numpy(network_input).float().div(255).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        detector(network_input)

    feature_map = capture.features[0]
    detections = [{"bbox": bbox} for bbox in frame_rows["bbox"]]
    rois, box_features = prepare_distance_inputs(
        detections, image.shape, feature_map, capture.strides[0]
    )
    target = torch.tensor(
        frame_rows["distance_m"].to_numpy(), device=DEVICE, dtype=torch.float32
    ).log()
    return feature_map, rois, box_features, target


sample_features, _, _, _ = load_frame_features(train_frames[0])
head = DistanceRegressionHead(feature_channels=sample_features.shape[1]).to(DEVICE)
optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=SCHEDULER_PATIENCE,
    threshold=MIN_IMPROVEMENT_M,
    threshold_mode="abs",
    min_lr=MIN_LEARNING_RATE,
)
loss_function = nn.SmoothL1Loss()
detector_sha256 = sha256_file(DETECTOR_WEIGHTS)

resume_path = LAST_WEIGHTS if LAST_WEIGHTS.exists() else BEST_WEIGHTS
resume_checkpoint = load_torch_checkpoint(resume_path)
if resume_checkpoint.get("detector_sha256") != detector_sha256:
    raise ValueError("The checkpoint was trained with different YOLO detector weights")
if resume_checkpoint["feature_channels"] != head.feature_channels:
    raise ValueError("The checkpoint feature size does not match the YOLO backbone")

head.load_state_dict(resume_checkpoint["head_state_dict"])
legacy_best_epoch = 0
if "optimizer_state_dict" in resume_checkpoint:
    optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in resume_checkpoint:
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    start_epoch = int(resume_checkpoint["epoch"])
    resume_mode = "exact checkpoint"
else:
    logged_epoch, legacy_best_epoch = tensorboard_training_state()
    start_epoch = int(
        resume_checkpoint.get("training_completed_epoch") or logged_epoch
    )
    if start_epoch == 0:
        raise RuntimeError("Could not determine the completed Colab epoch from TensorBoard")
    resume_mode = "best Colab weights with a new optimizer"

best_metrics = resume_checkpoint.get("metrics", {})
best_mae = float(
    resume_checkpoint.get("best_mae", best_metrics.get("mae_m", float("inf")))
)
best_epoch = int(resume_checkpoint.get("best_epoch") or legacy_best_epoch)
early_stopping_best_mae = float(
    resume_checkpoint.get("early_stopping_best_mae", best_mae)
)
epochs_without_improvement = int(
    resume_checkpoint.get("epochs_without_improvement", 0)
)

print(f"Resume: {resume_path}")
print(f"Completed epochs: {start_epoch}")
print(f"Continuation: epoch {start_epoch + 1} through {TOTAL_EPOCHS}")
print(f"Mode: {resume_mode}")


def run_epoch(frame_list, training):
    head.train(training)
    losses = []
    errors = []

    for frame_id in tqdm(frame_list, leave=False, desc="Train" if training else "Validation"):
        feature_map, rois, box_features, y_log = load_frame_features(frame_id)
        y_pred_log = head(feature_map, rois, box_features)
        loss = loss_function(y_pred_log, y_log)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        errors.extend((y_pred_log.detach().exp() - y_log.exp()).cpu().tolist())
        losses.append(loss.item())

    errors = np.asarray(errors)
    return {
        "loss": float(np.mean(losses)),
        "mae_m": float(np.mean(np.abs(errors))),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
    }


def checkpoint_payload(metrics, epoch, include_optimizer):
    payload = {
        "format_version": 2,
        "target_mode": "direct",
        "feature_channels": head.feature_channels,
        "hidden_channels": head.hidden_channels,
        "roi_size": head.roi_size,
        "head_state_dict": head.state_dict(),
        "detector_weights": DETECTOR_WEIGHTS.name,
        "detector_sha256": detector_sha256,
        "image_size": IMAGE_SIZE,
        "classes": TARGET_CLASSES,
        "target": "KITTI object-center z",
        "metrics": metrics,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_mae": best_mae,
        "early_stopping_best_mae": early_stopping_best_mae,
        "epochs_without_improvement": epochs_without_improvement,
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(TENSORBOARD_DIR, purge_step=start_epoch + 1)
writer.add_text("configuration/device", torch.cuda.get_device_name(0))
writer.add_text("configuration/resume_mode", resume_mode)

last_train_metrics = {}
last_val_metrics = best_metrics
completed_epoch = start_epoch
training_end_epoch = TOTAL_EPOCHS
if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
    print("The resume checkpoint had already reached the early-stopping condition.")
    training_end_epoch = start_epoch

try:
    for epoch in range(start_epoch + 1, training_end_epoch + 1):
        random.shuffle(train_frames)
        train_metrics = run_epoch(train_frames, training=True)
        with torch.no_grad():
            val_metrics = run_epoch(val_frames, training=False)

        print(
            f"Epoch {epoch:03d}/{TOTAL_EPOCHS} | "
            f"train MAE {train_metrics['mae_m']:.2f} m | "
            f"val MAE {val_metrics['mae_m']:.2f} m"
        )
        for name, value in train_metrics.items():
            writer.add_scalar(f"train/{name}", value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f"validation/{name}", value, epoch)
        scheduler.step(val_metrics["mae_m"])
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.flush()

        is_best = val_metrics["mae_m"] < best_mae
        if is_best:
            best_mae = val_metrics["mae_m"]
            best_epoch = epoch

        if val_metrics["mae_m"] < early_stopping_best_mae - MIN_IMPROVEMENT_M:
            early_stopping_best_mae = val_metrics["mae_m"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if is_best:
            atomic_torch_save(
                checkpoint_payload(val_metrics, epoch, include_optimizer=False),
                BEST_WEIGHTS,
            )
        atomic_torch_save(
            checkpoint_payload(val_metrics, epoch, include_optimizer=True),
            LAST_WEIGHTS,
        )
        last_train_metrics = train_metrics
        last_val_metrics = val_metrics
        completed_epoch = epoch

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch}: no improvement of "
                f"{MIN_IMPROVEMENT_M:.2f} m for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break
finally:
    writer.close()

if not BEST_WEIGHTS.exists():
    raise RuntimeError("Training finished without a deployable best checkpoint")

best_checkpoint = load_torch_checkpoint(BEST_WEIGHTS)
best_checkpoint["training_completed_epoch"] = completed_epoch
best_checkpoint["best_epoch"] = best_epoch
best_checkpoint["best_mae"] = best_mae
atomic_torch_save(best_checkpoint, BEST_WEIGHTS)

summary = {
    "run_name": RUN_NAME,
    "resume_from_epoch": start_epoch,
    "training_completed_epoch": completed_epoch,
    "maximum_epochs": TOTAL_EPOCHS,
    "stopped_early": completed_epoch < TOTAL_EPOCHS,
    "best_epoch": best_epoch,
    "best_validation_mae_m": best_mae,
    "detector_weights": DETECTOR_WEIGHTS.name,
    "detector_sha256": detector_sha256,
    "train_frames": len(train_frames),
    "validation_frames": len(val_frames),
    "objects": len(records),
    "last_train": last_train_metrics,
    "last_validation": last_val_metrics,
}
METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Best weights:", BEST_WEIGHTS)
print("Resume checkpoint:", LAST_WEIGHTS)
print("Metrics:", METRICS_PATH)
print("TensorBoard:", TENSORBOARD_DIR)
