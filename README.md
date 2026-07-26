# SpeedrEye

Lightweight pedestrian and cyclist detection with per-object distance estimation.

The application uses one YOLO backbone. It does not run MiDaS or LiDAR at inference time. KITTI camera and LiDAR data are used only to train and validate the distance head.

## Distance Methods

- `geometry`: calibrated pinhole estimate with an optional learned correction. It works without distance weights and is the default baseline.
- `direct`: predicts metric distance from the feature map already produced by YOLO. Train `models/distance/direct_distance.pt` before using it.
- `none`: object detection without distance estimation.

Direct and geometry-guided implementations are separated under `src/pipeline/distance/`. Both reuse the YOLO feature pyramid; neither adds a second image backbone.

## Run

Activate the Conda environment and run a video:

```powershell
conda activate ai
python src/pipeline/main.py --video videos/peatones3.mp4 --distance-method geometry
```

After training the direct head:

```powershell
python src/pipeline/main.py --camera --distance-method direct
```

The display reports YOLO time, distance-method time, total frame time, and sustained FPS.

## Train Direct Distance

Use [notebooks/train_direct_distance.ipynb](notebooks/train_direct_distance.ipynb).

For the simplest setup, open it directly in Colab:

[Open the direct-distance notebook in Google Colab](https://colab.research.google.com/github/MimiKat-Fir/SpeedrEye/blob/feature/direct-geometry-distance/notebooks/train_direct_distance.ipynb)

The notebook:

1. reads KITTI images, labels, camera calibration, and optional Velodyne files;
2. keeps only `Pedestrian` and `Cyclist` objects;
3. projects LiDAR points into each object box and uses their median forward distance when enough points match;
4. falls back to KITTI camera-coordinate `z` when an object has too few LiDAR returns;
5. trains a small direct-distance head while keeping the YOLO backbone frozen;
6. evaluates distance MAE/RMSE and saves the weights under `models/distance/`.

In Colab, the core KITTI files and full training are enabled automatically. The large raw Velodyne archive remains optional through `DOWNLOAD_LIDAR`.

The final artifacts use project-relative paths on every computer:

- `models/distance/direct_distance.pt`: portable direct-distance weights;
- `models/distance/direct_distance_metrics.json`: validation and compatibility metadata;
- `tensorboard/direct_distance/<run name>/`: TensorBoard event files.

The notebook downloads a ZIP containing these artifacts. Its final optional cell can commit and push the trained model and TensorBoard run when `PUBLISH_TO_GITHUB = True` and a `GITHUB_TOKEN` Colab secret is configured. After publication, collaborators only need:

```powershell
git switch feature/direct-geometry-distance
git pull
python src/pipeline/main.py --camera --distance-method direct
```

## Layout

```text
configs/                 KITTI configuration
data/                    local training and test data
models/calibration/      camera calibration
models/distance/         learned distance-head weights
models/yolo/             YOLO detector weights
notebooks/               training notebooks
results/                 calibration and runtime outputs
src/pipeline/            application pipeline
videos/                  test videos
```
