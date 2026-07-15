from pathlib import Path

from ultralytics import YOLO


MODEL_WEIGHTS = "yolo26n.pt"
DATA_CONFIG = "kitti.yaml"
IMAGE_SIZE = 640
STAGE_SIZE = 10
FINAL_EPOCH = 100
PROJECT_DIR = Path("runs") / "speedereye"


def train_stage(weights: str | Path, stage_end_epoch: int):
    model = YOLO(str(weights))
    stage_name = f"kitti_yolo26_stage_{stage_end_epoch:03d}"

    return model.train(
        data=DATA_CONFIG,
        epochs=STAGE_SIZE,
        imgsz=IMAGE_SIZE,
        project=str(PROJECT_DIR),
        name=stage_name,
        exist_ok=True,
    )


def main():
    # Load a pretrained YOLO26 model.
    current_weights: str | Path = MODEL_WEIGHTS

    # Train on KITTI in 10-epoch stages up to epoch 100.
    # Ultralytics saves weights after every stage in:
    # runs/speedereye/<stage_name>/weights/last.pt and best.pt
    for stage_end_epoch in range(STAGE_SIZE, FINAL_EPOCH + STAGE_SIZE, STAGE_SIZE):
        results = train_stage(current_weights, stage_end_epoch)
        current_weights = Path(results.save_dir) / "weights" / "last.pt"
        print(f"Stage {stage_end_epoch} complete. Saved weights: {current_weights}")

    print("TensorBoard logs are under runs/speedereye")
    print("Start TensorBoard with: tensorboard --logdir runs/speedereye")


if __name__ == "__main__":
    main()

