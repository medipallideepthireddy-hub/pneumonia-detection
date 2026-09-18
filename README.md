# AI-Based Pneumonia Detection Using Attention-Enhanced CNN

An end-to-end full-stack medical deep learning web application that analyzes chest radiographs (X-rays) to detect and classify **Pneumonia** vs **Normal** lung conditions with high precision. The system integrates a Convolutional Neural Network (CNN) augmented with a dual **Channel & Spatial Attention Mechanism (CBAM)** and generates **Grad-CAM (Gradient-weighted Class Activation Mapping)** attention heatmaps to localize pathological lung consolidations.

---

## 1. Project Overview

Pneumonia is an acute respiratory infection that affects millions worldwide, particularly vulnerable populations including children, the elderly, and immunocompromised patients. Rapid and reliable radiographic screening is critical for early therapeutic intervention.

This system provides:
- **Instantaneous Binary Classification**: Distinguishes between Normal lung radiographs and Pneumonia infiltrates.
- **Attention-Enhanced Feature Learning**: Directs convolutional kernels to focus on lung fields while suppressing extraneous signals (such as rib cages, clavicles, thoracic spine, and image borders).
- **Visual Interpretability (Grad-CAM)**: Overlays transparent color heatmaps onto the radiograph, indicating regions that contributed most to the diagnostic prediction.
- **Clinician-Friendly Dashboard**: Built with pure HTML5, CSS3, and modern Vanilla JavaScript for zero-dependency frontend performance.

---

## 2. Features

- **Responsive Medical Dashboard**: Glassmorphic UI styled for clinical environments, fully responsive on desktop, tablet, and mobile.
- **Drag-and-Drop Radiograph Upload**: Intuitive drag-and-drop dropzone with instant client-side image preview and format validation.
- **Attention-Enhanced CNN Architecture**:
  - Multi-stage convolutional blocks with Batch Normalization and ReLU activations.
  - Channel Attention Module (reweights informative feature maps via global pooling and shared MLP).
  - Spatial Attention Module (identifies focal spatial zones via inter-channel pooling and 7×7 convolutions).
  - Deep feature extraction head with Global Average Pooling and Dropout regularization.
- **Explainable AI (Grad-CAM)**:
  - Backpropagates class gradients to the final post-attention convolutional layer.
  - Produces Jet colormap heatmaps blended over the original radiograph.
- **Asynchronous Analysis**: Non-blocking `fetch()` API integration displaying real-time scanner animations and animated confidence meters.
- **Graceful Initialization & Fallback**: Starts immediately out-of-the-box with an architecture-initialized model if training on full datasets is pending.

---

## 3. Technologies Used

- **Frontend**:
  - HTML5, CSS3, Modern Vanilla JavaScript (ES6+)
  - Google Fonts (`Plus Jakarta Sans`, `JetBrains Mono`)
  - No heavy frameworks required (pure lightweight code)
- **Backend**:
  - Python 3.10+ (tested on Python 3.13)
  - Flask (REST API & template rendering)
  - Werkzeug (secure file handling)
- **Deep Learning & Computer Vision**:
  - TensorFlow 2.16+ / Keras 3.x
  - NumPy (tensor operations)
  - Pillow (image processing)
  - OpenCV (`cv2` for Grad-CAM colormapping and image blending)
  - Scikit-learn & Matplotlib (metrics & visualization)

---

## 4. Folder Structure

```
pneumonia-detection/
│
├── app.py                     # Flask web server and prediction endpoints
├── train.py                   # Attention-Enhanced CNN architecture & training pipeline
├── gradcam.py                 # Grad-CAM attention heatmap generator
├── requirements.txt           # Python dependencies
├── README.md                  # Complete documentation
│
├── model/
│   └── pneumonia_attention_cnn.keras   # Saved trained Attention-Enhanced CNN model
│
├── templates/
│   └── index.html             # Diagnostic dashboard interface
│
├── static/
│   ├── css/
│   │   └── style.css          # Medical AI dark-themed stylesheet
│   ├── js/
│   │   └── script.js          # Client-side validation, drag-and-drop & AJAX
│   └── uploads/               # Temporary uploads and generated heatmaps
│
└── dataset/                   # Dataset folder layout
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── validation/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## 5. Dataset Preparation

The training script expects the widely recognized **Chest X-Ray Images (Pneumonia)** dataset (Kermany et al., Mendeley Data / Kaggle):

1. Download the dataset from [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
2. Extract the images into the `dataset/` directory according to this layout:
   ```
   dataset/
     ├── train/
     │     ├── NORMAL/      # Healthy chest radiographs
     │     └── PNEUMONIA/   # Bacterial & viral pneumonia radiographs
     ├── validation/
     │     ├── NORMAL/
     │     └── PNEUMONIA/
     └── test/
           ├── NORMAL/
           └── PNEUMONIA/
   ```
3. Supported image formats: `.jpeg`, `.jpg`, `.png`.

---

## 6. Python Environment Setup

On macOS (Apple Silicon M1/M2/M3/M4 or Intel):

1. Open your Terminal application.
2. Navigate to your project directory:
   ```bash
   cd "/Users/deepthi/Desktop/Pnemounia detection"
   ```
3. Create a clean Python virtual environment:
   ```bash
   python3 -m venv venv
   ```
4. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

---

## 7. Installing Dependencies

With the virtual environment active, install all required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Training the Model

### Option A: Build and Save Architecture Immediately (Init Mode)
If you want to run the web application right away before downloading the multi-gigabyte dataset:
```bash
python train.py --init-only
```
This compiles the Attention-Enhanced CNN and creates `model/pneumonia_attention_cnn.keras`.

### Option B: Full Model Training on Dataset
When you have placed radiographs in the `dataset/` folder:
```bash
python train.py --epochs 20 --batch-size 32 --lr 0.0001
```

**Training Highlights**:
- **Automatic Class Balancing**: Applies weighted loss to prevent bias toward majority classes.
- **Data Augmentation**: Applies random horizontal flips, mild rotations, zoom, and contrast adjustments.
- **Early Stopping & Checkpoints**: Automatically restores the weights with the lowest validation loss and saves the model directly to `model/pneumonia_attention_cnn.keras`.

---

## 9. Running the Flask Application

Launch the Flask web server:

```bash
python app.py
```

Console output will confirm the server is running:
```
[Model Loader] Loading trained model from .../model/pneumonia_attention_cnn.keras...
[Model Loader] Successfully loaded model.
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

---

## 10. Opening the Application in the Browser

Open your browser (Safari, Chrome, Firefox, or Edge) and navigate to:

```
http://127.0.0.1:5000
```

---

## 11. How Prediction Works

1. **Upload**: User drags and drops or browses for a chest radiograph (`.jpg`, `.jpeg`, `.png`).
2. **Client Validation**: Verifies extension and checks file size limit ($\le 16\text{ MB}$).
3. **Preprocessing**:
   - Resized to $224 \times 224$ pixels.
   - Converted to 3-channel RGB.
   - Normalized: $\text{pixels} = \frac{\text{pixel}}{255.0} \in [0.0, 1.0]$.
   - Batched to shape `(1, 224, 224, 3)`.
4. **Attention-Enhanced Inference**:
   - Model passes tensor through convolutional layers, channel attention, spatial attention, and classification head.
   - Evaluates sigmoid probability $P \in [0.0, 1.0]$.
   - Classification rule:
     - $P \ge 0.50 \implies \mathbf{PNEUMONIA}$, Confidence $= P \times 100\%$
     - $P < 0.50 \implies \mathbf{NORMAL}$, Confidence $= (1 - P) \times 100\%$
5. **Grad-CAM Visualization**:
   - Gradients of the predicted class score are calculated relative to `target_conv_layer`.
   - Feature maps are weighted and passed through a ReLU activation.
   - Heatmap is resized and blended over the radiograph using OpenCV (`cv2.COLORMAP_JET`).
6. **Result Presentation**:
   - Real-time animated confidence meter.
   - Clinical explanation summary.
   - Original radiograph side-by-side with the Attention Visualization.

---

## 12. Attention Mechanism in Medical Imaging

In standard CNNs, all receptive fields contribute equally to subsequent layers, which can cause networks to erroneously correlate non-pulmonary landmarks (e.g., patient positioning clips, rib bone contours, or diaphragmatic angles) with pathology.

Our **Dual CBAM Attention Block** solves this:
1. **Channel Attention (What to look for)**:
   $$\mathbf{M}_c(\mathbf{F}) = \sigma(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F})))$$
   Identifies which semantic feature filters encode consolidation textures rather than background anatomical structure.
2. **Spatial Attention (Where to look)**:
   $$\mathbf{M}_s(\mathbf{F}) = \sigma(f^{7 \times 7}([\text{AvgPool}(\mathbf{F}); \text{MaxPool}(\mathbf{F})]))$$
   Assigns high weights to alveolar airspace consolidation zones and suppresses healthy tissue or non-lung background.

---

## 13. System Limitations

- **Image Quality Dependence**: Over-exposed, rotated, or cropped non-standard radiographs may reduce diagnostic accuracy.
- **Specific Pathologies**: Designed specifically to differentiate healthy lungs from pneumonia infiltrates; it is not calibrated to diagnose pneumothorax, cardiomegaly, or nodules without retrained multi-label heads.
- **Dataset Bias**: Diagnostic performance reflects the demographic and scanner distribution of the training dataset.

---

## 14. Medical Disclaimer

> **IMPORTANT**: This application is developed strictly for **academic, educational, and scientific research purposes**. It does not constitute medical advice, a clinical diagnosis, or a certified medical device. Healthcare decisions must never be made solely on predictions generated by this software. Always consult a licensed medical professional or radiologist for clinical evaluation.

---

## 15. Quick Reference: Step-by-Step Terminal Commands

Run these exact commands in your macOS Terminal:

```bash
# 1. Navigate to the project root directory
cd "/Users/deepthi/Desktop/Pnemounia detection"

# 2. Create the virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Build and save the Attention-Enhanced CNN model
python train.py --init-only

# (Optional: If dataset images are present, train for 20 epochs)
# python train.py --epochs 20

# 6. Start the Flask server
python app.py

# 7. Open in your web browser
open http://127.0.0.1:5000
```
