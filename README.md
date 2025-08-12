# damageClassification

Read the [Final Report](./2021-11-30_ece9603_projectReport.pdf) for details about the repository.


# CNN-Based Building Damage Classification

## Overview
Wind-induced damage to buildings is one of the primary sources of hazard-related loss. Processing post-disaster survey imagery of affected areas is important to quantify the incurred loss and inform future building resilience improvements.  

This repository implements a **Convolutional Neural Network (CNN)**–based deep learning pipeline to classify building damage levels from post-hurricane imagery. The dataset is derived from publicly available data collected after four hurricanes in the US.  

Two CNN architectures are available:
- **CNNBN** – CNN with batch normalization layers
- **CNN** – CNN without batch normalization  

The models are trained and evaluated using **multi-class cross-entropy loss**, with **F1-score** and **balanced accuracy** as performance metrics.

Our best-performing model achieved:
- **F1-score**: 0.741
- **Balanced accuracy**: 63.9%

These results are competitive compared to previous studies using the same dataset.

---

## Repository Structure
``` yaml
.
├── create_input_data.py # Prepares dataset into train/validation/test folders
├── train_classifier.py # Trains, validates, and tests the CNN models
├── models
|   └── models.py # CNN and CNNBN model architectures in PyTorch
├── output/ # Saved training and evaluation metrics
└── data/ # Expected folder for processed images
```


---

## Installation
1. **Clone this repository**
   ```bash
   git clone https://github.com/Tsinuel/damageClassification.git
   cd damageClassification-main
   ```

## Install dependencies
  ```bash
  pip install torch torchvision scikit-learn matplotlib pandas numpy
  ```

# Dataset Preparation

## Obtain the dataset
This project uses xBD-derived hurricane imagery datasets, stored with associated .csv label files.

## Organize raw data
Ensure your files follow the structure expected in:
  - **../xBD_csv/** containing **train.csv** and **test.csv**

