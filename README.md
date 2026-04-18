
# Microscopic Image Classification: Chlorella Detection 🔬🦠

## 📌 Overview
This project is a PyTorch-based Deep Learning pipeline designed for a Neural Networks Competition. The primary objective is to classify microscopic images into 5 distinct categories, with a specific emphasis on accurately identifying the target class: **'Chlorella'**.

## 🚧 Challenges
The competition presented several unique constraints that required a tailored approach:
* **Data Constraints:** A small dataset consisting of grayscale, unstructured biological images.
* **Class Imbalance:** A high volume of background noise ('Debris') compared to a very low volume of the target 'Chlorella' algae.
* **Strict Scoring Metric:** The optimization target is Precision, but it carries a strict constraint: Recall must remain above 0.5 (50%).

## 🧠 AI Architecture & Pipeline

### 1. Data Strategy
* **Merging Modalities:** We utilized all available image types (Amplitude, Phase, Mask) to increase data volume and improve model robustness.
* **Channel Adaptation:** The input images are 1-channel grayscale, but our chosen ResNet backbone expects 3-channel RGB inputs. We duplicated the grayscale channel 3x to preserve the pre-trained weights.
* **Augmentation:** To prevent overfitting on biological data, we apply random resizing/cropping, horizontal/vertical flips, and a crucial 180° random rotation via `torchvision`.

### 2. Model: Transfer Learning
* **Backbone:** We use a lightweight **ResNet18** pre-trained on ImageNet, which is deep enough for texture analysis without being overly complex.
* **Modifications:** The Fully Connected head was modified from 1000 classes to 5.
* **Optimization:** AdamW optimizer (for better weight decay handling) and a `ReduceLROnPlateau` scheduler for fine-tuning.

### 3. Handling Class Imbalance
We implemented a **Weighted CrossEntropyLoss** to heavily penalize the model for missing the rare target class. We calculated inverse frequency weights for all classes and manually doubled (2x) the weight specifically for 'Chlorella'.

### 4. Custom Evaluation & Post-Processing
Since standard accuracy does not align with the competition's rules, we built a custom pipeline:
* **Custom Validation Loop:** Tracks the specific competition metric (`Score = Precision` if `Recall > 0.5`). The best model (`best_model.pth`) is saved *only* when this custom score improves.
* **Threshold Optimization:** The default decision boundary of 0.5 is mathematically suboptimal here. A post-training script scans boundaries between 0.05 and 0.95 to find the perfect Precision/Recall trade-off.
* **Test Time Augmentation (TTA):** During inference, each image is predicted 3 times (Original, Flipped Horizontally, Flipped Vertically). The probabilities are averaged to ensure highly stable predictions.

## 🚀 Setup & Execution

### Prerequisites
* Python 3.8+
* PyTorch & Torchvision
* Pillow, NumPy

### Folder Structure
Ensure your dataset is organized as follows before running:

.
└── microscopic-classification/
    ├── main.py/
    │   └── best_model.pth
    ├── submission.csv
    ├── README.md
    ├── train/
    │   ├── class_chlorella/
    │   ├── class_debris/
    │   ├── class_haematococcus/
    │   ├── class_small_haemato/
    │   └── class_small_particle/
    └── test/
        ├── 001.png
        ├── 002.png
        └── ... 
        
## Running the Pipeline

To train the model, optimize the threshold, and generate predictions, simply run:

Bash
python main.py
The script will automatically output a submission.csv containing the final predictions, ready for the competition.
