from pathlib import Path

DEFAULT_WEIGHTS = Path("weights") / "speedereye_epoch_001.pt"
TEST_SOURCE = Path("data") / "test"
OUTPUT_DIR = Path("outputs")


def main():
    if not DEFAULT_WEIGHTS.exists():
        print(f"Missing weights: {DEFAULT_WEIGHTS}")
        print("Run the first training notebook cell to create them.")
        return

    from ultralytics import YOLO

    model = YOLO(str(DEFAULT_WEIGHTS))
    print(f"Loaded weights: {DEFAULT_WEIGHTS}")

    if TEST_SOURCE.exists() and any(TEST_SOURCE.iterdir()):
        results = model.predict(
            source=str(TEST_SOURCE),
            device="cpu",
            project=str(OUTPUT_DIR),
            name="predict",
            exist_ok=True,
            save=True,
        )
        print(f"Predictions: {len(results)}")
    else:
        print(f"Put test images or a video in {TEST_SOURCE} to run CPU prediction.")


main()

