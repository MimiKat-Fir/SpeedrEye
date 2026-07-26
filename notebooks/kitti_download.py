"""
Descarga (si hace falta) el dataset KITTI "object detection" directamente
desde el bucket oficial de KITTI, sin necesidad de login manual, y lo deja
en <project_root>/data/kitti_raw con la estructura estándar:

    data/kitti_raw/training/image_2/*.png
    data/kitti_raw/training/label_2/*.txt
    data/kitti_raw/training/calib/*.txt
    data/kitti_raw/testing/...

Se usa tanto desde train_detector.py como desde train_distance_head.py.
"""

import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

KITTI_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
KITTI_FILES = [
    "data_object_image_2.zip",   # ~12.6 GB — imágenes
    "data_object_label_2.zip",   # ~5.6 MB  — etiquetas 2D/3D
    "data_object_calib.zip",     # ~5 MB    — calibración de cámara (necesaria para el script de distancia)
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KITTI_DIR = PROJECT_ROOT / "data" / "kitti_raw"


def _download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r   {dest_path.name}: {pct:5.1f}% ({downloaded/1e6:.0f}/{total/1e6:.0f} MB)",
                          end="", flush=True)
    print()


def ensure_kitti_downloaded(kitti_dir: Path = DEFAULT_KITTI_DIR, force: bool = False) -> Path:
    """
    Se asegura de que kitti_dir contenga training/image_2, training/label_2
    y training/calib. Si falta algo, descarga y extrae los zips oficiales
    de KITTI ahí mismo. Devuelve kitti_dir.
    """
    kitti_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = kitti_dir / "_downloads"
    downloads_dir.mkdir(exist_ok=True)

    markers = {
        "data_object_image_2.zip": kitti_dir / "training" / "image_2",
        "data_object_label_2.zip": kitti_dir / "training" / "label_2",
        "data_object_calib.zip": kitti_dir / "training" / "calib",
    }

    for filename in KITTI_FILES:
        target_dir = markers[filename]
        already_present = target_dir.exists() and any(target_dir.iterdir())

        if already_present and not force:
            print(f"✅ {filename}: ya presente en {target_dir}, se omite descarga")
            continue

        zip_path = downloads_dir / filename
        if not zip_path.exists() or force:
            url = KITTI_BASE_URL + filename
            print(f"⬇️  Descargando {filename} desde {url}")
            try:
                _download_file(url, zip_path)
            except Exception as e:
                raise RuntimeError(
                    f"No se pudo descargar {filename} desde {url}. "
                    f"Si tu red bloquea el bucket de KITTI, descárgalo manualmente desde "
                    f"https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d "
                    f"y colócalo en {zip_path}."
                ) from e

        print(f"📦 Extrayendo {filename} en {kitti_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(kitti_dir)

    return kitti_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Descarga KITTI object detection en data/kitti_raw")
    parser.add_argument("--dest", type=Path, default=DEFAULT_KITTI_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_kitti_downloaded(args.dest, force=args.force)
    print(f"\n✅ KITTI listo en: {args.dest}")
