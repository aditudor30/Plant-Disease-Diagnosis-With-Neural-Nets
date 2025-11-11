# Plant Disease Diagnosis With Neural Nets

## Overview

This project focuses on diagnosing plant diseases using deep learning. It compares a **custom-built Convolutional Neural Network (CNN)** with a **fine-tuned ResNet18** model to identify and classify plant leaf diseases from images.

Both models were trained and evaluated on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle.

---

## Dataset

- **Source:** [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Classes:** 38
- **Structure:**
  - `train/` — training images
  - `valid/` — validation images
- **Total images:** ~87,000

Each subfolder corresponds to a specific crop and disease (e.g., `Apple___Black_rot`, `Tomato___Late_blight`, etc.).

---

## Project Structure

```
Plant-Disease-Diagnosis-With-Neural-Nets/
│
├── main_notebook.ipynb       # Jupyter notebook with all training and evaluation steps
├── requirements.txt           # Python dependencies
├── models/                    # Saved model weights
├── data/                      # Dataset (train/valid directories)
└── README.md                  # Project documentation
```

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/aditudor30/Plant-Disease-Diagnosis-With-Neural-Nets.git
cd Plant-Disease-Diagnosis-With-Neural-Nets
pip install -r requirements.txt
```

---

## Model Architectures

### 1. Custom CNN

A lightweight convolutional model built from scratch using **PyTorch**.

**Architecture Summary:**
- 3 Convolutional layers (ReLU activations + MaxPooling)
- 2 Fully Connected layers
- Dropout regularization
- Softmax output for 38 classes

**Pros:**
- Simpler and faster to train
- Lower computational cost

**Cons:**
- Limited feature extraction capability
- Lower accuracy on complex leaf textures

---

### 2. Fine-tuned ResNet18

A pre-trained **ResNet18** model (from ImageNet) was fine-tuned on the plant disease dataset.

**Modifications:**
- Final fully connected layer replaced with a 38-class classifier
- Trained with data augmentation and transfer learning

**Pros:**
- High accuracy due to deep residual learning
- Benefits from pre-trained feature extraction

**Cons:**
- Requires more GPU memory and longer training time

---

## Training Details

| Parameter | Custom CNN | ResNet18 |
|------------|-------------|----------|
| Epochs | 10 | 5 |
| Batch Size | 32 | 32 |
| Optimizer | Adam | Adam |
| Learning Rate | 0.001 | 0.0001 (fine-tuning) |
| Loss Function | CrossEntropyLoss | CrossEntropyLoss |

---

## Results

| Model | Validation Accuracy | Validation Loss |
|--------|----------------------|-----------------|
| Custom CNN | ~96.71% | ~0.11 |
| ResNet18 (fine-tuned) | ~99.67% | ~0.0118 |

**Observation:**  
ResNet18 significantly outperformed the custom CNN in both accuracy and generalization, showing the effectiveness of transfer learning in visual classification tasks.

---

## Usage

After training, you can run inference on a single image:

```python
from PIL import Image
from torchvision import transforms
import torch

# Load model
model = torch.load('models/resnet18_finetuned.pth')
model.eval()

# Transform input
img = Image.open('sample_leaf.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Predict
input_tensor = transform(img).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
pred_class = torch.argmax(output, 1).item()
print("Predicted class:", pred_class)
```

---

## Future Improvements

- Experiment with deeper models (ResNet50, EfficientNet)
- Add explainability using Grad-CAM
- Implement web interface for real-time diagnosis
- Expand dataset for more crop species

---

## Acknowledgments

- [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- PyTorch and torchvision for model development
- Open source community for continuous support

---
