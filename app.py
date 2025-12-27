# ================================================================
#  Respiratory Sound Analysis 
# ================================================================
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

import os
import io
import json
import math
import tempfile
from glob import glob

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

import soundfile as sf
import scipy.signal

import cv2

import tensorflow as tf  # <--- use this instead of from tensorflow.keras.models import load_model

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI(api_key=OPENAI_KEY)

def generate_explanation(pred_class, probs):
    prompt = f"""
You are a medical AI assistant.

A deep learning model analyzed lung sound audio and produced the following result:

Predicted condition: {pred_class}
Class probabilities:
- none: {probs[0]*100:.1f}%
- crackles: {probs[1]*100:.1f}%
- wheezes: {probs[2]*100:.1f}%
- both: {probs[3]*100:.1f}%

Explain to a patient in clear, simple language:
1. What this result likely means
2. Whether it is potentially concerning
3. What actions they should consider
4. That this is not a medical diagnosis
Keep it short and friendly.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content


# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Respiratory Sound Analysis",
    layout="wide",
)

st.markdown("""
<style>
/* Main containers */
.block-container {
    padding-top: 2rem;
}

/* Card look */
div[data-testid="stMetric"] {
    background-color: #0f172a;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 0 12px rgba(34,197,94,0.25);
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 17px;
}

/* Section titles */
h1, h2, h3 {
    color: #e5e7eb;
}

/* Success / warning / error */
div[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)



st.title("🫁 Respire AI")

# ------------------------------
# USER-EDITABLE PATHS
# ------------------------------
DEFAULT_DATASET_PATH = r"C:\Users\hp\Desktop\Biomed\Respiratory_Sound_Database"

with st.sidebar:
    st.header("⚙️ Settings")
    DATASET_PATH = st.text_input("Dataset path", value=DEFAULT_DATASET_PATH)
    AUDIO_PATH = os.path.join(DATASET_PATH, "audio_and_txt_files")

    MODEL_PATH = st.text_input("Model file", value="respiratory_sound_model_second.keras")

    st.write("---")
    st.markdown("**Classes** (must match training):")
    CLASS_NAMES = ['none', 'crackles', 'wheezes', 'both']
    st.write(CLASS_NAMES)








# ------------------------------
# LOAD GROUND TRUTH LABELS
# ------------------------------
@st.cache_data
def load_ground_truth(audio_dir):
    gt = {}
    for txt in glob(os.path.join(audio_dir, "*.txt")):
        base = os.path.basename(txt).replace(".txt", "")
        data = pd.read_csv(txt, sep="\t", header=None)
        data.columns = ["start", "end", "crackles", "wheezes"]

        has_c = data["crackles"].any()
        has_w = data["wheezes"].any()

        if has_c and has_w:
            gt[base] = "both"
        elif has_c:
            gt[base] = "crackles"
        elif has_w:
            gt[base] = "wheezes"
        else:
            gt[base] = "none"
    return gt

GROUND_TRUTH = load_ground_truth(AUDIO_PATH)


# ------------------------------
# LOAD DEMO POOL (NEW)
# ------------------------------
DEMOPOOL_PATH = "demopool.json"

if not os.path.exists(DEMOPOOL_PATH):
    st.error("demopool.json not found. Please run build_demopool.py first.")
    DEMO_FILES = []
else:
    with open(DEMOPOOL_PATH, "r") as f:
        DEMO_FILES = json.load(f)

# ------------------------------
# CACHED LOADERS
# ------------------------------
@st.cache_data
def list_audio_files(audio_dir):
    if not os.path.exists(audio_dir):
        return []
    files = sorted(glob(os.path.join(audio_dir, "*.wav")))
    return files

# ------------------------------
# SIGNAL PROCESSING FUNCTIONS
# (Copied & simplified from your training code so shapes match)
# ------------------------------

def bitrate_channels(lp_wave):
    bps = (lp_wave.getsampwidth() / lp_wave.getnchannels())
    return (bps, lp_wave.getnchannels())

def read24bitwave(lp_wave):
    nFrames = lp_wave.getnframes()
    buf = lp_wave.readframes(nFrames)
    reshaped = np.frombuffer(buf, np.int8).reshape(nFrames, -1)
    short_output = np.empty((nFrames, 2), dtype=np.int8)
    short_output[:, :] = reshaped[:, -2:]
    short_output = short_output.view(np.int16)
    return (lp_wave.getframerate(), np.divide(short_output, 32768).reshape(-1))

def extract2FloatArr(wavfile_obj, filename):
    import scipy.io.wavfile as wf
    (bps, channels) = bitrate_channels(wavfile_obj)

    if bps in [1, 2, 4]:
        (rate, data) = wf.read(filename)
        divisor_dict = {1: 255, 2: 32768}
        if bps in [1, 2]:
            divisor = divisor_dict[bps]
            data = np.divide(data, float(divisor))
        return (rate, data)

    elif bps == 3:
        return read24bitwave(wavfile_obj)

    else:
        raise Exception('Unrecognized wave format: {} bytes per sample'.format(bps))


def resample(current_rate, data, target_rate):
    x_original = np.linspace(0, 100, len(data))
    x_resampled = np.linspace(0, 100, int(len(data) * (target_rate / current_rate)))
    resampled = np.interp(x_resampled, x_original, data)
    return (target_rate, resampled.astype(np.float32))


def read_wav_file(path, target_rate=22000):
    import wave
    import librosa

    # Load audio with librosa for robust handling
    data, sample_rate = librosa.load(path, sr=None, mono=True)

    # 🔹 Trim leading & trailing silence
    data, _ = librosa.effects.trim(data, top_db=25)

    # 🔹 Normalize amplitude to [-1, 1]
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak

    # 🔹 Resample if needed
    if sample_rate != target_rate:
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_rate)
        sample_rate = target_rate

    return (sample_rate, data.astype(np.float32))




def generate_padded_samples(source, output_length):
    # Normalize
    if np.max(np.abs(source)) > 0:
        source = source / np.max(np.abs(source))

    # Trim silence
    energy = np.abs(source)
    threshold = 0.02 * np.max(energy)
    idx = np.where(energy > threshold)[0]
    if len(idx) > 0:
        source = source[idx[0]:idx[-1]+1]

    # Pad / trim
    out = np.zeros(output_length, dtype=np.float32)
    if len(source) >= output_length:
        out[:] = source[:output_length]
    else:
        out[:len(source)] = source

    return out





def Freq2Mel(freq):
    return 1125 * np.log(1 + freq / 700)


def Mel2Freq(mel):
    exponents = mel / 1125
    return 700 * (np.exp(exponents) - 1)


def GenerateMelFilterBanks(mel_space_freq, fft_bin_frequencies):
    n_filters = len(mel_space_freq) - 2
    coeff = []
    for mel_index in range(n_filters):
        m = int(mel_index + 1)
        filter_bank = []
        for f in fft_bin_frequencies:
            if (f < mel_space_freq[m - 1]):
                hm = 0
            elif (f < mel_space_freq[m]):
                hm = (f - mel_space_freq[m - 1]) / (mel_space_freq[m] - mel_space_freq[m - 1])
            elif (f < mel_space_freq[m + 1]):
                hm = (mel_space_freq[m + 1] - f) / (mel_space_freq[m + 1] - mel_space_freq[m])
            else:
                hm = 0
            filter_bank.append(hm)
        coeff.append(filter_bank)
    return np.array(coeff, dtype=np.float32)


def FFT2MelSpectrogram(f, Sxx, sample_rate, n_filterbanks, vtlp_params=None):
    (max_mel, min_mel) = (Freq2Mel(max(f)), Freq2Mel(min(f)))
    mel_bins = np.linspace(min_mel, max_mel, num=(n_filterbanks + 2))
    mel_freq = Mel2Freq(mel_bins)

    if vtlp_params is None:
        filter_banks = GenerateMelFilterBanks(mel_freq, f)
    else:
        # In this app we won’t use VTLP; keep for compatibility
        (alpha, f_high) = vtlp_params
        warp_factor = min(alpha, 1)
        nyquist_f = sample_rate / 2
        threshold_freq = f_high * warp_factor / alpha
        warped_mel = np.where(mel_freq <= threshold_freq,
                              mel_freq * alpha,
                              nyquist_f - (nyquist_f - mel_freq) *
                              ((nyquist_f - f_high * warp_factor) /
                               (nyquist_f - f_high * (warp_factor / alpha))))
        filter_banks = GenerateMelFilterBanks(warped_mel, f)

    mel_spectrum = np.matmul(filter_banks, Sxx)
    return (mel_freq[1:-1], np.log10(mel_spectrum + float(10e-12)))


def sample2MelSpectrum_single(audio_chunk, sample_rate=22000, n_filters=50):
    """
    Simplified version just for inference:
    - Takes a 1D audio chunk (already padded to 5 seconds)
    - Returns normalized mel spectrogram of shape (n_filters, time, 1)
    """
    n_rows = 175
    n_window = 512
    f, t, Sxx = scipy.signal.spectrogram(
        audio_chunk,
        fs=sample_rate,
        nfft=n_window,
        nperseg=n_window
    )
    Sxx = Sxx[:n_rows, :].astype(np.float32)
    _, mel_log = FFT2MelSpectrogram(f[:n_rows], Sxx, sample_rate, n_filterbanks=n_filters)
    mel_min = np.min(mel_log)
    mel_max = np.max(mel_log)
    diff = mel_max - mel_min
    if diff > 0:
        norm_mel_log = (mel_log - mel_min) / diff
    else:
        norm_mel_log = np.zeros(shape=(n_filters, Sxx.shape[1]))
    return np.reshape(norm_mel_log, (n_filters, Sxx.shape[1], 1)).astype(np.float32)


def preprocess_wav_for_model(path, target_rate=22000, desired_length=5.0):
    """
    Full preprocessing chain consistent with training:
    - Load & resample
    - Pad/trim to desired_length seconds
    - Compute mel spectrogram (50 x T x 1)
    - Return batch tensor (1, 50, T, 1)
    """
    sample_rate, data = read_wav_file(path, target_rate)
    output_length = int(desired_length * sample_rate)
    padded = generate_padded_samples(data, output_length)
    mel = sample2MelSpectrum_single(padded, sample_rate=sample_rate, n_filters=50)
    # shape: (50, T, 1) -> add batch dim
    return np.expand_dims(mel, axis=0), mel  # (1, H, W, 1), (H, W, 1)

def occlusion_map(model, mel, class_index, patch_size=6):
    mel_map = mel[:, :, 0]
    heatmap = np.zeros_like(mel_map)

    baseline_pred = model.predict(np.expand_dims(mel, 0))[0][class_index]

    for i in range(0, mel_map.shape[0], patch_size):
        for j in range(0, mel_map.shape[1], patch_size):
            occluded = mel_map.copy()
            occluded[i:i+patch_size, j:j+patch_size] = 0

            occluded_input = np.expand_dims(occluded[..., None], 0)
            pred = model.predict(occluded_input)[0][class_index]

            heatmap[i:i+patch_size, j:j+patch_size] = baseline_pred - pred

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    return heatmap

def overlay_occlusion(mel, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (mel.shape[1], mel.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    mel_img = np.uint8(255 * mel[:, :, 0])
    mel_img = cv2.cvtColor(mel_img, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(mel_img, 1 - alpha, heatmap, alpha, 0)


# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_cnn_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

model = load_cnn_model(MODEL_PATH)


LAST_CONV_LAYER = "conv2d_7"

def load_waveform(path, target_rate=None):
    """
    Simple loader for plotting / playback
    """
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if target_rate is not None and sr != target_rate:
        from librosa import resample
        y = resample(y, orig_sr=sr, target_sr=target_rate)
        sr = target_rate
    return y, sr

# ------------------------------
# VISUALIZATION HELPERS
# ------------------------------

def plot_waveform(y, sr, title="Waveform"):
    fig, ax = plt.subplots(figsize=(8, 3))
    t = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(t, y)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(mel, title="Mel Spectrogram"):
    """
    mel: (H, W, 1)
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(mel[:, :, 0], aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel filter index")
    fig.tight_layout()
    return fig


def plot_probs(probs, class_names):
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(class_names, probs, edgecolor="black")
    ax.set_ylim([0, 1])
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f"{p*100:.1f}%", ha="center")
    ax.set_ylabel("Probability")
    ax.set_title("Predicted Class Probabilities")
    fig.tight_layout()
    return fig


# ------------------------------
# LOAD MODEL
# ------------------------------
model = load_cnn_model(MODEL_PATH)

# ------------------------------
# TABS
# ------------------------------
tabs = st.tabs([
    "Overview",
    "EDA & Features",
    "Model Performance",
    "Demo – Test Sample",
    "Demo –  Upload Your Audio",
])

# ==============================
# TAB 1: OVERVIEW
# ==============================
with tabs[0]:
    st.subheader("📚 Project Overview")

    st.markdown("""
## 🫁 Respire AI - An Intelligent Respiratory Sound Analysis System

This application presents a complete **end-to-end biomedical signal processing and deep learning pipeline** for the **automatic detection of respiratory abnormalities** from lung sound recordings.

The system is designed to simulate a real clinical decision-support workflow — from raw biosignal acquisition to interpretable diagnostic output.

---

### 📁 Dataset
**Respiratory Sound Database (Kaggle)**  
• **920** lung sound recordings  
• **126** unique patients  
• Cycle-level annotations for:
  - **Crackles**
  - **Wheezes**

Each respiratory cycle is labeled by clinical experts, enabling fine-grained pathological analysis.

---

### 🧪 Diagnostic Classes
1. **none** — Normal breathing (no crackles, no wheezes)  
2. **crackles** — Presence of crackling sounds only  
3. **wheezes** — Presence of wheezing sounds only  
4. **both** — Simultaneous crackles and wheezes  

---

### 🔬 Biomedical & Machine Learning Pipeline

This project implements a full research-grade processing pipeline:

#### 🧭 Exploratory Data Analysis (EDA)
- Class & diagnosis distribution
- Chest location analysis
- Respiratory cycle statistics
- Dataset balance & demographics

#### 🔊 Signal Processing & Feature Engineering
- Raw waveform & spectrogram visualization
- Mel spectrogram generation
- Time-domain features: RMS, energy, ZCR
- Frequency-domain features: MFCCs, spectral centroid, bandwidth, roll-off
- Statistical descriptors: skewness, kurtosis, variance

#### 📐 Preprocessing & Statistical Validation
- Resampling & amplitude normalization
- Silence trimming & padding
- Normality testing (Shapiro–Wilk)
- Q–Q plots
- Scaling comparison (Standard, MinMax, Robust)

#### 🧬 Feature Selection & Biomedical Interpretation
- MANOVA + univariate ANOVA
- Mutual information
- Recursive Feature Elimination (RFE)
- Random Forest importance
- Intersection-based feature selection for robust biomarkers

#### 🧠 Deep Learning Model
- CNN trained on normalized Mel spectrograms
- Extensive data augmentation:
  - Time-stretching
  - Vocal Tract Length Perturbation (VTLP)
  - FFT rolling
- Subject-wise train/test splitting for clinically realistic evaluation

#### 📊 Model Evaluation & Explainability
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Per-class performance visualization
- Sample-level prediction inspection with confidence scores
- Clinical interpretation of results

---

### 🧪 How to Explore
Use the tabs above to navigate through:
- **EDA & Features** → dataset insights and biomedical analysis  
- **Model Performance** → training curves, metrics, and predictions  
- **Live Demo** → test on dataset samples or your **own recorded breathing sounds**

---

### 👨‍💻 Project Contributors
**Saleha (231209)**  
**Zeyyan (231221)**  
**Talha (231223)**  
**Isfah (231207)**

This project demonstrates how modern AI can assist clinicians in early respiratory disorder screening while maintaining interpretability and biomedical rigor.
""")


# ==============================
# TAB 2: EDA & FEATURES
# ==============================
with tabs[1]:
    st.subheader("📊 Exploratory Data Analysis & Features")

    col1, col2 = st.columns(2)

    # Show saved EDA images if they exist
    eda_images = [
        ("eda_characteristics.png", "EDA Characteristics Overview"),
        ("duration_by_class_boxplot.png", "Duration by Class Boxplot"),
        ("duration_vs_cycles_scatter.png", "Duration vs Respiratory Cycles"),
        ("02_diagnosis_distribution.png", "Diagnosis Distribution"),
        ("03_chest_location_distribution.png", "Chest Location Distribution"),
        ("05_class_diagnosis_heatmap.png", "Class vs Diagnosis Heatmap"),
        ("raw_vs_preprocessed_waveform.png", "Raw vs Preprocessed Lung Sound"),
        ("spectrogram_vs_mel.png", "Spectrogram vs Mel Spectrogram"),
        ("pca_before_after_preprocessing.png", "PCA Before vs After Preprocessing"),
        ("correlation_before_after_feature_selection.png", "Correlation Before vs After Feature Selection"),
        ("boxplot_before_after_standardization.png", "Feature Scaling Effect"),
        ("distribution_shift_preprocessing.png", "Distribution Shift After Preprocessing"),
        ("feature_normalization_effect.png", "Feature Normalization Effect"),
        ("all_features_before_after.png", "All Features Before vs After Processing"),
        ("07_shapiro_wilk_test.png", "Shapiro–Wilk Normality Test"),
        ("09_qq_plots.png", "Q–Q Plots for Feature Normality"),
        ("08_feature_distributions.png", "Feature Distributions by Class"),
        ("10_scaling_comparison.png", "Feature Scaling Comparison"),
        ("11_anova_results.png", "ANOVA Results for Features"),
        ("12_correlation_matrix.png", "Feature Correlation Matrix"),
        ("13_feature_importance.png", "Feature Importance from Random Forest"),
        ("14_mel_spectograms_by_class.png", "Mel Spectrograms by Class"),
        ("15_train_test_split.png", "Train-Test Split Visualization"),
        
    ]

    for idx, (fname, title) in enumerate(eda_images):
        path = f"assets/eda/{fname}"
        if os.path.exists(path):
            with (col1 if idx % 2 == 0 else col2):
                st.markdown(f"**{title}**")
                st.image(path, width="stretch")

    st.markdown("---")

    # Show feature tables if available
    col3, col4 = st.columns(2)

    with col3:
        if os.path.exists("extracted_features.csv"):
            st.markdown("**Extracted Features (sample)**")
            feat_df = pd.read_csv("extracted_features.csv")
            st.dataframe(feat_df.head(20))
        else:
            st.info("`extracted_features.csv` not found – add it in the app folder to show feature table.")

    with col4:
        if os.path.exists("selected_features.csv"):
            st.markdown("**Selected Features (after feature selection)**")
            sel_df = pd.read_csv("selected_features.csv")
            st.dataframe(sel_df)
        else:
            st.info("`selected_features.csv` not found – add it in the app folder to show selected features.")


# ==============================
# TAB 3: MODEL PERFORMANCE
# ==============================
with tabs[2]:
    st.subheader("📈 Model Training & Evaluation")
    
        # Sample predictions visualization
    if os.path.exists("assets/model/sample_predictions_second.png"):
        st.markdown("---")
        st.subheader("🔍 Sample Model Predictions")

        colA, colB = st.columns([1, 2])

        with colA:
            st.markdown("""
            **What this shows**

            Each image represents a test sample's **Mel spectrogram** with:

            • True class  
            • Predicted class  
            • Model confidence  

            🟢 **Green** = correct prediction  
            🔴 **Red** = incorrect prediction
            """)

        with colB:
            st.image(
                "assets/model/sample_predictions_second.png",
                caption="Sample Predictions — Green = Correct | Red = Incorrect",
                width="stretch"
            )
    st.markdown("---")

    col1, col2 = st.columns(2)

    # Show saved training curves if available
    train_curve = "assets/model/training_history_second.png"
    if os.path.exists(train_curve):
        with col1:
            st.markdown("**Training & Validation Curves**")
            st.image(train_curve, width="stretch")

    # Confusion matrix
    cm_path = "assets/model/confusion_matrix_second.png"
    if os.path.exists(cm_path):
        with col2:
            st.markdown("**Confusion Matrix**")
            st.image(cm_path, width="stretch")

    st.markdown("---")

    col3, col4 = st.columns(2)

    per_class_path = "assets/model/per_class_metrics_second.png"
    if os.path.exists(per_class_path):
        with col3:
            st.markdown("**Per-Class Precision / Recall / F1**")
            st.image(per_class_path, width="stretch")

    dist_path = "assets/model/label_distribution_second.png"
    if os.path.exists(dist_path):
        with col4:
            st.markdown("**True vs Predicted Label Distribution**")
            st.image(dist_path, width="stretch")

    # Raw metrics JSON if available
    if os.path.exists("evaluation_metrics_second.json"):
        st.markdown("---")
        st.subheader("📊 Detailed Evaluation Metrics")

        with open("evaluation_metrics_second.json", "r") as f:
            metrics = json.load(f)

        # --- Summary ---
        st.metric("Overall Accuracy", f"{metrics['overall_accuracy']*100:.2f}%")

        # --- Per-class table ---
        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
            "Support": metrics["support"]
        })

        st.markdown("### 🧪 Per-Class Performance")
        st.dataframe(
            df.style.format({
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 Score": "{:.3f}"
            }),
            width="stretch"
        )

        st.markdown("### 🧬 Clinical Interpretation")

        interpretation_text = f"""
        **Overall Model Performance**

        The model achieved an overall classification accuracy of **{metrics['overall_accuracy']*100:.2f}%**, 
        indicating strong capability in identifying respiratory sound patterns across the four clinical categories.

        **Per-Class Analysis**

        - **Normal (None):** High precision and recall indicate reliable identification of healthy breathing patterns with minimal false alarms.
        - **Crackles:** Balanced precision and recall suggest consistent detection of crackling sounds often associated with pulmonary disorders such as pneumonia or fibrosis.
        - **Wheezes:** Moderate performance reflects the acoustic complexity of wheezing sounds, which often overlap with other respiratory phenomena.
        - **Both:** Slightly lower scores are expected due to overlapping acoustic features when crackles and wheezes occur simultaneously.

        **Clinical Implication**

        This performance level suggests that the system is suitable as a **clinical decision-support tool**, 
        providing early screening assistance to clinicians by highlighting potentially abnormal lung sounds.  
        However, final diagnosis should always remain with a trained healthcare professional.
        """

        st.info(interpretation_text)



# ==============================
# TAB 4: DEMO – TEST SAMPLE
# ==============================
with tabs[3]:
    st.subheader("🎧 Demo on Dataset Test Sample")

    if not os.path.exists(AUDIO_PATH):
        st.error(f"Audio path not found: {AUDIO_PATH}")
    elif model is None:
        st.error("Model not loaded. Check the model path in the sidebar.")
    else:
        audio_files = DEMO_FILES
        if not audio_files:
            st.warning("No .wav files found in the audio directory.")
        else:
            selected_file = st.selectbox(
                "Choose a recording from the dataset",
                audio_files,
                format_func=lambda p: os.path.basename(p),
            )

            if selected_file:
                st.markdown(f"**Selected file:** `{os.path.basename(selected_file)}`")

                # Load waveform for playback & plotting
                y, sr = load_waveform(selected_file)

                # Audio player
                st.audio(selected_file, format="audio/wav")

                # Compute model input
                with st.spinner("Preprocessing & predicting..."):
                    X_batch, mel = preprocess_wav_for_model(selected_file)
                    preds = model.predict(X_batch)[0]
                    pred_idx = int(np.argmax(preds))
                    pred_class = CLASS_NAMES[pred_idx]
                    
                    
                with st.spinner("Generating medical explanation..."):
                    explanation = generate_explanation(pred_class, preds)

                st.markdown("### 🧠 AI Medical Explanation")
                st.info(explanation)

                true_class = GROUND_TRUTH.get(os.path.basename(selected_file).replace(".wav",""), "Unknown")

                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.markdown(f"### 🧠 Model Prediction: **{pred_class}**")
                with col_info2:
                    st.markdown(f"### 🏷 Ground Truth: **{true_class}**")

                if pred_class == true_class:
                    st.success("✅ Correct Prediction")
                else:
                    st.error("❌ Incorrect Prediction")

                col_plot1, col_plot2 = st.columns(2)

                with col_plot1:
                    st.pyplot(plot_waveform(y, sr, title="Waveform"))

                with col_plot2:
                    st.pyplot(plot_mel_spectrogram(mel, title="Mel Spectrogram (Model Input)"))

                # ---------- XAI: Occlusion Sensitivity ----------
                with st.spinner("Generating Explainable AI visualization..."):
                    heatmap = occlusion_map(model, mel, pred_idx)
                    overlay = overlay_occlusion(mel, heatmap)

                st.markdown("### 🔬 Explainable AI – Model Attention")
                st.image(overlay, caption="Occlusion Map: Red = most influential regions", width="stretch")
                # ----------------------------------------------


                st.markdown("#### Class Probabilities")
                st.pyplot(plot_probs(preds, CLASS_NAMES))


# ==============================
# TAB 5: DEMO – UPLOAD ONLY
# ==============================

with tabs[4]:
    st.subheader("📤 Demo with Your Own Audio")

    if model is None:
        st.error("Model not loaded. Check the model path in the sidebar.")
    else:
        uploaded = st.file_uploader(
            "Upload a WAV file (record breathing using any recorder and upload here)",
            type=["wav"]
        )

        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded.read())
                temp_wav_path = tmp.name

            st.markdown("---")
            st.markdown("### Processing your audio...")

            y, sr = load_waveform(temp_wav_path)
            st.audio(temp_wav_path, format="audio/wav")

            with st.spinner("Preprocessing & predicting..."):
                X_batch, mel = preprocess_wav_for_model(temp_wav_path)
                preds = model.predict(X_batch)[0]
                pred_idx = int(np.argmax(preds))
                pred_class = CLASS_NAMES[pred_idx]

            
            with st.spinner("Generating medical explanation..."):
                explanation = generate_explanation(pred_class, preds)

            st.markdown("### 🧠 AI Medical Explanation")
            st.info(explanation)


            st.markdown(f"### 🔍 Predicted class: **{pred_class}**")

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_waveform(y, sr, title="Waveform"))
            with col2:
                st.pyplot(plot_mel_spectrogram(mel, title="Mel Spectrogram (Model Input)"))


            st.markdown("#### Class Probabilities")
            st.pyplot(plot_probs(preds, CLASS_NAMES))
            


