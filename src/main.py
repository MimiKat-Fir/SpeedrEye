import os
from ultralytics import YOLO
from src.config.paths import RAW_DATA_DIR

def download_kitti():
    """
    Descarga automáticamente el dataset KITTI usando Ultralytics.
    Los datos se guardarán en: ./datasets/kitti/
    """
    print("=== SpeedrEye - Descarga del Dataset KITTI ===")
    print("Iniciando descarga... (esto puede tomar varios minutos)")
    
    # Este paso descarga y prepara el dataset automáticamente
    # La primera vez que se ejecuta, descarga el archivo kitti.zip (~390 MB)
    model = YOLO("yolo11n.pt")  # Modelo ligero solo para activar la descarga
    model.train(data="kitti.yaml", epochs=1, imgsz=640, workers=0)
    
    print("\n✅ ¡Descarga completada!")
    print(f"Los datos se encuentran en: {os.path.abspath('./datasets/kitti')}")
    print("\nEstructura de datos descargada:")
    print("  datasets/kitti/")
    print("  ├── images/")
    print("  │   ├── train/  (5,985 imágenes)")
    print("  │   └── val/    (1,496 imágenes)")
    print("  └── labels/")
    print("      ├── train/  (5,985 archivos de etiquetas)")
    print("      └── val/    (1,496 archivos de etiquetas)")

if __name__ == "__main__":
    download_kitti()