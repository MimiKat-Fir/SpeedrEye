#!/usr/bin/env python3
"""
Paso 1 del pipeline SpeedrEye: entrena el detector YOLO SOLO con imágenes.

- Convierte las etiquetas de KITTI (formato label_2) a formato YOLO,
  filtrando ÚNICAMENTE Pedestrian y Cyclist (todo lo demás se descarta:
  Car, Van, Truck, Tram, Misc, DontCare, Person_sitting).
- No toca LIDAR ni calibración: solo imágenes + bboxes 2D.
- Entrena YOLO26n (o el modelo que elijas) con esos datos.

Uso:
    python train_detector.py \
        --kitti-root /ruta/a/KITTI/object_detection \
        --output-dir ./kitti_yolo \
        --model yolo26n.pt \
        --imgsz 640 \
        --epochs 150

Estructura esperada de KITTI (la típica del "object detection" KITTI):
    <kitti-root>/training/image_2/000000.png ...
    <kitti-root>/training/label_2/000000.txt ...
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_download import ensure_kitti_downloaded, DEFAULT_KITTI_DIR, PROJECT_ROOT  # noqa: E402

# ============================================================
# Mapeo de clases KITTI -> SpeedrEye
# ============================================================
# Person_sitting se EXCLUYE a propósito: la geometría de distancia
# (altura_real * focal / altura_bbox) asume persona de pie, y mezclar
# posturas sentadas metería ruido en el detector para poco beneficio,
# dado que en tu caso de uso (cámara frontal en movimiento) es rara.
KITTI_TO_YOLO_CLASS = {
    "Pedestrian": 0,
    "Cyclist": 1,
}
CLASS_NAMES = {0: "Pedestrian", 1: "Cyclist"}


def parse_kitti_label(label_path):
    """Lee un fichero label_2 de KITTI y devuelve una lista de objetos crudos."""
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 15:
                continue
            obj_type = parts[0]
            bbox = tuple(map(float, parts[4:8]))  # x1, y1, x2, y2
            objects.append({"type": obj_type, "bbox": bbox})
    return objects


def convert_kitti_to_yolo(kitti_root: Path, output_dir: Path, val_ratio: float = 0.15, seed: int = 42):
    """
    Convierte KITTI (solo Pedestrian/Cyclist) a la estructura que espera
    Ultralytics: images/{train,val} + labels/{train,val} + dataset.yaml
    """
    images_dir = kitti_root / "training" / "image_2"
    labels_dir = kitti_root / "training" / "label_2"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"No encuentro {images_dir} o {labels_dir}. "
            "Revisa --kitti-root (debe apuntar a la carpeta que contiene 'training/')."
        )

    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No hay imágenes .png en {images_dir}")

    print(f"📦 {len(image_files)} imágenes encontradas en KITTI")

    # Split por bloques contiguos (no aleatorio puro) para reducir fuga de
    # información entre train/val, ya que frames consecutivos de KITTI
    # suelen pertenecer al mismo trayecto y son casi idénticos entre sí.
    random.seed(seed)
    n = len(image_files)
    block_size = 10
    blocks = [image_files[i:i + block_size] for i in range(0, n, block_size)]
    random.shuffle(blocks)

    n_val_blocks = max(1, int(len(blocks) * val_ratio))
    val_blocks = blocks[:n_val_blocks]
    train_blocks = blocks[n_val_blocks:]

    train_files = [f for block in train_blocks for f in block]
    val_files = [f for block in val_blocks for f in block]

    print(f"   Train: {len(train_files)} | Val: {len(val_files)}")

    kept_boxes = 0
    dropped_images = 0

    for split_name, files in (("train", train_files), ("val", val_files)):
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            objects = parse_kitti_label(label_path)
            relevant = [o for o in objects if o["type"] in KITTI_TO_YOLO_CLASS]

            if not relevant:
                dropped_images += 1
                continue  # sin peatón/ciclista, no aporta al detector

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            yolo_lines = []
            for obj in relevant:
                x1, y1, x2, y2 = obj["bbox"]
                x1, x2 = max(0, x1), min(img_w, x2)
                y1, y2 = max(0, y1), min(img_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_id = KITTI_TO_YOLO_CLASS[obj["type"]]
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                kept_boxes += 1

            if not yolo_lines:
                dropped_images += 1
                continue

            shutil.copy(img_path, img_out / img_path.name)
            with open(lbl_out / f"{img_path.stem}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

    print(f"✅ Cajas conservadas (Pedestrian+Cyclist): {kept_boxes}")
    print(f"⏭️  Imágenes sin objetos relevantes (descartadas): {dropped_images}")

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {output_dir.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n"
            f"  0: Pedestrian\n"
            f"  1: Cyclist\n"
        )
    print(f"📝 dataset.yaml escrito en {yaml_path}")
    return yaml_path


def train(dataset_yaml: Path, model_name: str, imgsz: int, epochs: int, batch, output_dir: Path,
          models_dir: Path, output_name: str):
    from ultralytics import YOLO

    model = YOLO(model_name)  # pesos preentrenados en COCO como punto de partida
    model.train(
        data=str(dataset_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        project=str(output_dir),
        name="speedreye_detector",
        # Clases muy desbalanceadas / pequeñas en escena: estos ajustes
        # ayudan a que el detector no ignore objetos lejanos o poco frecuentes.
        patience=30,
        cos_lr=True,
        close_mosaic=15,  # desactiva mosaic los últimos N epochs, mejora precisión final
        degrees=0.0,      # poca rotación tiene sentido, cámara frontal fija
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        # Dataset en disco externo/USB: cachear en RAM evita relecturas lentas
        # repetidas cada época (visto en el log: "Slow image access detected").
        cache="ram",
        workers=4,
    )

    best_weights = output_dir / "speedreye_detector" / "weights" / "best.pt"
    print(f"\n✅ Entrenamiento del detector terminado. Pesos crudos en: {best_weights}")

    models_dir.mkdir(parents=True, exist_ok=True)
    final_path = models_dir / output_name
    shutil.copy(best_weights, final_path)
    print(f"📦 Copiado a: {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Entrena el detector SpeedrEye (solo clases) con KITTI")
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_DIR,
                         help=f"Por defecto: {DEFAULT_KITTI_DIR} (se descarga solo si falta)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "kitti_yolo")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "yolo")
    parser.add_argument("--output-name", type=str, default="speedreye_kitti_v2.pt",
                         help="Nombre del fichero final en --models-dir (no sobrescribe modelos previos)")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="yolo26n.pt o yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=str, default="-1",
                         help="Entero fijo, o -1 para que Ultralytics elija automáticamente "
                              "según la VRAM libre (recomendado en GPUs pequeñas como la tuya)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--skip-conversion", action="store_true",
                         help="Usa esto si dataset.yaml ya existe en --output-dir")
    parser.add_argument("--no-download", action="store_true",
                         help="No intentar descargar KITTI aunque falte en --kitti-root")
    args = parser.parse_args()
    batch = int(args.batch)  # acepta "-1" (auto) o un entero positivo

    if not args.no_download and not args.skip_conversion:
        ensure_kitti_downloaded(args.kitti_root)

    if args.skip_conversion:
        dataset_yaml = args.output_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"No existe {dataset_yaml}, quita --skip-conversion")
    else:
        dataset_yaml = convert_kitti_to_yolo(args.kitti_root, args.output_dir, args.val_ratio)

    train(dataset_yaml, args.model, args.imgsz, args.epochs, batch, args.output_dir,
          args.models_dir, args.output_name)


if __name__ == "__main__":
    main()
