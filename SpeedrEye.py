from pathlib import Path

DEFAULT_WEIGHTS = Path("weights") / "speedereye_epoch_001.pt"
TEST_IMAGE = "test.jpg"


def main():
    if not DEFAULT_WEIGHTS.exists():
        print(f"Missing weights: {DEFAULT_WEIGHTS}")
        print("Run the first training notebook cell to create them.")
        return

    from ultralytics import YOLO

    model = YOLO(str(DEFAULT_WEIGHTS))
    print(f"Loaded weights: {DEFAULT_WEIGHTS}")

    if Path(TEST_IMAGE).exists():
        results = model.predict(source=TEST_IMAGE, device="cpu", save=True)
        print(f"Predictions: {len(results)}")
    else:
        print(f"Put a test image at {TEST_IMAGE} to run a CPU prediction.")


if __name__ == "__main__":
    main()

