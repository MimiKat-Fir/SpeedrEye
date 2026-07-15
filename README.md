# SpeedrEye

AI project about LiDAR and vision for autonomous cars.

The current goal is to process images, including fine-tuning work, and finish with an in-real-life demo using a mobile phone.

More requirements, dataset details, model choices, and demo constraints will be added as the project is defined.

## Training Notebook

Use the notebook for training:

- [notebooks/train_model.ipynb](notebooks/train_model.ipynb)
- CPU is forced for now with `device="cpu"`.
- The first cell runs 1 epoch as a smoke test.
- Weights are saved under `weights/` and kept in Git for collaboration.
- TensorBoard logs are saved under `tensorboard/` and kept in Git for collaboration.

TensorBoard:

```powershell
tensorboard --logdir tensorboard
```

## Check Script

Use `SpeedrEye.py` only to load weights and test a prediction:

```powershell
conda activate ai
python SpeedrEye.py
```

Put test images or a video in `data/test/`. Prediction outputs are written to `outputs/predict/`.
