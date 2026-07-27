#!/usr/bin/env python3
"""
Paso 2 del pipeline SpeedrEye - VERSIÓN V2
Entrena la cabecita de corrección de distancia para el detector V3.
Usa los datos KITTI ya descargados en data/kitti_raw/
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Añadir src al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline.distance.head import (
    BackboneFeatureCapture,
    DistanceRegressionHead,
    prepare_distance_inputs,
    file_sha256,
)

# RUTAS FIJAS (usando data/ de la raíz)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KITTI_ROOT = PROJECT_ROOT / "data" / "kitti_raw"

CLASS_HEIGHTS = {0: 1.70, 1: 1.70}
KITTI_TO_YOLO_CLASS = {"Pedestrian": 0, "Cyclist": 1}
MIN_DISTANCE, MAX_DISTANCE = 0.5, 60.0


def parse_calib(calib_path):
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                values = list(map(float, line.strip().split(" ")[1:]))
                p2 = np.array(values).reshape(3, 4)
                return {"fx": p2[0, 0], "fy": p2[1, 1], "cx": p2[0, 2], "cy": p2[1, 2]}
    raise ValueError(f"No se encontró P2 en {calib_path}")


def parse_kitti_full_label(label_path):
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 15:
                continue
            obj_type = parts[0]
            if obj_type not in KITTI_TO_YOLO_CLASS:
                continue
            bbox = tuple(map(float, parts[4:8]))
            location = tuple(map(float, parts[11:14]))
            objects.append({
                "class": KITTI_TO_YOLO_CLASS[obj_type],
                "bbox": bbox,
                "z_meters": location[2],
            })
    return objects


class KittiDistanceDataset(Dataset):
    def __init__(self, kitti_root: Path, file_ids):
        self.kitti_root = kitti_root
        self.file_ids = file_ids

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        img_path = self.kitti_root / "training" / "image_2" / f"{file_id}.png"
        label_path = self.kitti_root / "training" / "label_2" / f"{file_id}.txt"
        calib_path = self.kitti_root / "training" / "calib" / f"{file_id}.txt"

        image = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]
        objects = parse_kitti_full_label(label_path)
        calib = parse_calib(calib_path)
        return image, objects, calib


def collate_single(batch):
    return batch[0]


def compute_targets(objects, calib, image_shape):
    dets, targets = [], []
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        box_h_px = max(y2 - y1, 1.0)
        real_height = CLASS_HEIGHTS[obj["class"]]

        dist_geom = real_height * calib["fx"] / box_h_px
        dist_geom = min(max(dist_geom, MIN_DISTANCE), MAX_DISTANCE)

        dist_real = obj["z_meters"]
        if dist_real <= 0:
            continue

        factor = dist_real / dist_geom
        factor = min(max(factor, 0.2), 5.0)

        dets.append({"bbox": (x1, y1, x2, y2), "class": obj["class"]})
        targets.append(np.log(factor))

    return dets, torch.tensor(targets, dtype=torch.float32)


def get_file_ids(kitti_root: Path, val_ratio: float, seed: int = 42):
    label_dir = kitti_root / "training" / "label_2"
    ids = sorted(p.stem for p in label_dir.glob("*.txt"))

    random.seed(seed)
    block_size = 10
    blocks = [ids[i:i + block_size] for i in range(0, len(ids), block_size)]
    random.shuffle(blocks)
    n_val = max(1, int(len(blocks) * val_ratio))
    val_ids = [i for b in blocks[:n_val] for i in b]
    train_ids = [i for b in blocks[n_val:] for i in b]
    return train_ids, val_ids


def run_epoch(loader, detector_model, capture, head, optimizer, device, train_mode: bool):
    head.train(train_mode)
    total_loss, n_batches = 0.0, 0

    for image, objects, calib in loader:
        if not objects:
            continue

        dets, targets = compute_targets(objects, calib, image.shape)
        if not dets:
            continue
        targets = targets.to(device)

        with torch.no_grad():
            detector_model.predict(image, verbose=False, conf=0.001, imgsz=640)

        if not capture.features:
            continue
        feature_map = capture.features[0].to(device)

        rois, box_features = prepare_distance_inputs(dets, image.shape, feature_map, capture.strides[0])
        rois, box_features = rois.to(device), box_features.to(device)

        if train_mode:
            optimizer.zero_grad()
            preds = head(feature_map, rois, box_features)
            loss = nn.functional.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = head(feature_map, rois, box_features)
                loss = nn.functional.mse_loss(preds, targets)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Entrena cabecita de distancia para detector V3")
    parser.add_argument("--kitti-root", type=Path, default=KITTI_ROOT)
    parser.add_argument("--detector-weights", type=Path, required=True,
                         help="Pesos del detector V3 (models/yolo/speedreye_kitti_v3.pt)")
    parser.add_argument("--output", type=Path, 
                         default=PROJECT_ROOT / "models" / "distance" / "geometry_guided_v2.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--roi-size", type=int, default=3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    args = parser.parse_args()

    if not args.kitti_root.exists():
        print(f"❌ No se encuentra KITTI en: {args.kitti_root}")
        print("   Asegúrate de que data/kitti_raw/training/ existe")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    from ultralytics import YOLO

    print("📦 Cargando detector congelado...")
    detector = YOLO(str(args.detector_weights))
    detector.model.to(device)
    for p in detector.model.parameters():
        p.requires_grad_(False)
    detector.model.eval()

    capture = BackboneFeatureCapture(detector.model)

    dummy = np.zeros((384, 1280, 3), dtype=np.uint8)
    with torch.no_grad():
        detector.predict(dummy, verbose=False, conf=0.001, imgsz=640)
    feature_channels = capture.features[0].shape[1]
    print(f"   Canales de feature map: {feature_channels}")

    head = DistanceRegressionHead(
        feature_channels=feature_channels,
        hidden_channels=args.hidden_channels,
        roi_size=args.roi_size,
    ).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    train_ids, val_ids = get_file_ids(args.kitti_root, args.val_ratio)
    print(f"   Train: {len(train_ids)} | Val: {len(val_ids)}")

    train_loader = DataLoader(
        KittiDistanceDataset(args.kitti_root, train_ids),
        batch_size=1, shuffle=True, collate_fn=collate_single, num_workers=2,
    )
    val_loader = DataLoader(
        KittiDistanceDataset(args.kitti_root, val_ids),
        batch_size=1, shuffle=False, collate_fn=collate_single, num_workers=2,
    )

    best_val_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, detector, capture, head, optimizer, device, train_mode=True)
        val_loss = run_epoch(val_loader, detector, capture, head, optimizer, device, train_mode=False)
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "target_mode": "geometry_guided",
                "feature_channels": feature_channels,
                "hidden_channels": args.hidden_channels,
                "roi_size": args.roi_size,
                "head_state_dict": head.state_dict(),
                "detector_weights": args.detector_weights.name,
                "detector_sha256": file_sha256(args.detector_weights),
            }
            torch.save(checkpoint, args.output)
            print(f"   💾 Checkpoint guardado en {args.output} (val_loss={val_loss:.4f})")

    print(f"\n✅ Entrenamiento terminado. Mejor val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()