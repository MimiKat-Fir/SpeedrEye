# SpeedrEye

AI project about LiDAR and vision for autonomous cars.

The current goal is to process images, including fine-tuning work, and finish with an in-real-life demo using a mobile phone.

More requirements, dataset details, model choices, and demo constraints will be added as the project is defined.

## Training

Local environment:

```powershell
conda activate ai
python SpeedrEye.py
```

The script trains in 10-epoch stages up to 100 epochs. Ultralytics saves weights after each stage under `runs/speedereye/.../weights/`.

TensorBoard:

```powershell
tensorboard --logdir runs/speedereye
```

Training is expected to run on Google Colab for GPU access. Keep the dataset close to the training runtime: use local/Drive storage in Colab instead of repeatedly pulling data over the network during training.
