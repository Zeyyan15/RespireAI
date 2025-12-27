

# 🫁 **Respiratory Sound Analysis & Deep Learning Diagnosis System**

> **Automatic Detection of Crackles & Wheezes from Lung Sound Recordings using Biomedical Signal Processing and CNNs**

---

## 📌 **Project Overview**

This project presents a **complete biomedical signal processing and deep learning pipeline** for automated detection of respiratory abnormalities — **Crackles, Wheezes, Both, or Normal** — from lung sound recordings.

The system follows **state-of-the-art biomedical practices**, including:

* Medical-grade signal preprocessing
* Statistical feature analysis
* Rigorous feature selection
* Deep learning with augmentation
* Clinical evaluation & interpretability
* A full interactive web demo

---

## 🧬 **Dataset**

**Respiratory Sound Database (Kaggle)**

* **920 recordings**
* **126 patients**
* Cycle-level annotations
* TXT files marking:

  * Start & end of each breathing cycle
  * Presence of crackles & wheezes

---

## 🧪 **Biomedical Signal Processing Pipeline**

### 1️⃣ **Raw Audio Ingestion**

* Multi-format WAV loading (8-bit, 16-bit, 24-bit, 32-bit)
* Automatic conversion to **float32**
* Unified sampling rate: **22,000 Hz**

### 2️⃣ **Noise Handling & Normalization**

* **Resampling** → uniform sampling across all patients
* **Amplitude normalization** → signals scaled to stable numeric range
* **Silence trimming** → removes recording artifacts and background noise
* **Dynamic padding & trimming** → fixed 5-second segments

> This guarantees consistent physiological input for the learning system.

---

## 🫀 **Respiratory Cycle Extraction**

Each recording is split using medical annotations:

```
(start_time, end_time, crackles, wheezes)
```

Each cycle is isolated using:

* Time-accurate slicing
* Physiological segmentation
* Label mapping → **one-hot encoding**

---

## 🎛️ **Time–Frequency Feature Engineering**

### 🔹 Spectrogram → Mel Spectrogram

* STFT window: **512**
* 175 frequency bins
* **50 Mel filter banks**
* Log-scaled power spectrum
* Min-Max normalization across time–frequency space

### 🔹 Voice Tract Length Perturbation (VTLP)

Simulates physiological variation across patients:

* Random vocal tract warping
* Frequency axis distortion
* Mimics inter-subject lung acoustics

### 🔹 Biomedical Augmentation

| Technique       | Purpose                             |
| --------------- | ----------------------------------- |
| Time Stretching | Simulates breathing speed variation |
| VTLP            | Models anatomical variability       |
| FFT Rolling     | Introduces phase invariance         |
| Segment slicing | Improves generalization             |

---

## 📊 **Statistical & Classical Feature Analysis**

The project includes:

* Time-domain features
* Frequency-domain features
* MFCCs, Chroma, Tonnetz
* Spectral centroid, roll-off, RMS, ZCR
* Higher-order stats: **skewness, kurtosis**

### 🧠 Biomedical Statistics

* **Shapiro–Wilk normality testing**
* **Q–Q plots**
* **MANOVA**
* **Univariate ANOVA**
* Feature scaling comparison:

  * StandardScaler
  * MinMaxScaler
  * RobustScaler

### 🧬 Feature Selection

* ANOVA F-test
* Mutual Information
* Recursive Feature Elimination
* Random Forest importance
* **Intersection selection for optimal biomarkers**

---

## 🧠 **Deep Learning Architecture**

### CNN Model

```
Input → Conv → Conv → Conv → Deep Residual Blocks → Dense → Softmax
```

* Multi-scale convolution kernels
* LeakyReLU activations
* Dropout regularization
* Adam optimizer
* Categorical cross-entropy loss

### Training Strategy

* **Subject-wise splitting** (no patient leakage)
* Balanced sampling across all clinical classes
* Heavy augmentation for rare classes
* 25 epochs | Batch size 128

---

## 🧪 **Evaluation & Clinical Metrics**

| Metric                  | Implemented |
| ----------------------- | ----------- |
| Accuracy                | ✅           |
| Precision               | ✅           |
| Recall                  | ✅           |
| F1-Score                | ✅           |
| Confusion Matrix        | ✅           |
| Per-class metrics       | ✅           |
| Clinical interpretation | ✅           |

Includes:

* Confusion matrix visualization
* Class-wise performance plots
* Sample prediction inspection
* Probability distribution analysis

---

## 🖥️ **Interactive Web Application**

Built using **Streamlit**:

* Dataset demo with ground-truth vs prediction
* Upload your own lung sounds
* Visualization of:

  * Waveform
  * Mel spectrogram
  * Prediction probabilities
* Medical-themed UI
* Session accuracy tracking
* Explainable AI clinical text output

---

## 🧾 **Scientific Alignment**

This pipeline aligns closely with modern biomedical research toolkits and best practices:

| Biomedical Standard           | Your System |
| ----------------------------- | ----------- |
| PhysioNet-style preprocessing | ✅           |
| Medical signal normalization  | ✅           |
| Statistical inference         | ✅           |
| Feature selection             | ✅           |
| Deep learning diagnostics     | ✅           |
| Clinical interpretability     | ✅           |
| Subject-wise validation       | ✅           |

---

## 🏁 **Conclusion**

This project demonstrates a **complete end-to-end biomedical diagnostic system** — from raw lung sound recordings to clinically interpretable predictions — following the same principles used in real hospital research environments.

It bridges:

> **Biomedical signal processing + statistical analysis + deep learning + clinical interpretability + user-facing medical software**

