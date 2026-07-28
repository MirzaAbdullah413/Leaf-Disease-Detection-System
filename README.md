## 🌿 Leaf Disease Detection
A convolutional neural network that classifies plant leaf diseases from images, trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage). 
Includes Grad-CAM visualization to highlight which regions of the leaf drove the prediction, so you can sanity-check *why* the model made a call, not just *what* it predicted.
## Demo
Given a leaf image, the model predicts the disease class and overlays a heatmap showing the regions it focused on:
```
Predicted class: Tomato___Late_blight
```
<img width="323" height="316" alt="image" src="https://github.com/user-attachments/assets/d5aaa543-45b7-48ae-88e4-aa73223c0a29" />
## Model Architecture
A custom CNN built from 5 stacked convolutional blocks, followed by a fully connected classifier:

- **5× Conv Blocks**, each: `Conv2d → BatchNorm2d → ReLU → MaxPool2d`
  - Channels: 3 → 32 → 64 → 128 → 256 → 512
- **Classifier**: `Flatten → Linear(512×7×7 → 512) → ReLU → Dropout(0.5) → Linear(512 → num_classes)`
Input images are resized to `224×224` and normalized using the dataset's computed per-channel mean/std.
~14.4M trainable parameters, with the vast majority (~89%) concentrated in the first classifier layer.

## Dataset
Trained on [PlantVillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) — leaf images across multiple crop species (tomato, potato, pepper, etc.) with healthy and diseased classes, pre-split into `train/` and `val/` folders.

## Repository Structure
```
├── Train_Model.py   # Training script — run on Google Colab (GPU)
├── Predict_Image.py       # Inference script — run locally/offline on any machine
└── README.md
```
## Setup
### 1. Training (Google Colab)
1. Upload `PlantVillage.zip` (containing `PlantVillage/train` and `PlantVillage/val`) to your Google Drive under `MyDrive/`.
2. Open `Train_Model.py` in Google Colab.
3. Run it — it will:
   - Mount your Google Drive
   - Extract the dataset locally into the Colab runtime for faster I/O
   - Compute dataset mean/std for normalization
   - Train the CNN for 20 epochs, tracking the best validation accuracy
   - Save the **full trained model** (architecture + weights + class names + normalization stats) to:
     ```
     /content/drive/MyDrive/leaf_disease_model/leaf_disease_model.pth
     ```

> ⚠️ Training on the full dataset for 20 epochs can take a while on Colab's free GPU tier — keep the tab active to avoid disconnects before the save completes.

### 2. Inference (locally, offline)
Once training finishes, download `leaf_disease_model.pth` from your Google Drive to your laptop.
**One-time setup:**
```bash
pip install torch torchvision opencv-python pillow matplotlib numpy
```

**Run a prediction:**
```bash
python Predict_Image.py path/to/leaf_disease_model.pth path/to/leaf_image.jpg
```

This prints the predicted class and displays a Grad-CAM heatmap overlaid on the original image, indicating the regions that most influenced the prediction. No GPU, internet, or dataset access is required at inference time — it works fully offline from the saved `.pth` file.

## Limitations

- **Domain shift**: the model is trained on lab-style images (plain backgrounds, controlled lighting). Real-world photos (different backgrounds, lighting, or camera angles) can reduce accuracy.
- **Visually similar diseases**: some diseases across different crops (e.g. potato and tomato late blight) are caused by the same pathogen and look nearly identical, which can lead to cross-class confusion.
- **Class imbalance**: some crops/classes have far more training images than others, which can bias predictions toward better-represented classes.

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
