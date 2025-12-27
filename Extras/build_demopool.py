import os, json
from glob import glob
import numpy as np
import soundfile as sf
import scipy.signal
import tensorflow as tf

DATASET_PATH = r"C:\Users\hp\Desktop\Biomed\Respiratory_Sound_Database"
AUDIO_PATH = os.path.join(DATASET_PATH, "audio_and_txt_files")
MODEL_PATH = "respiratory_sound_model_second.keras"
CLASS_NAMES = ['none','crackles','wheezes','both']

def generate_padded_samples(source, length):
    out = np.zeros(length, dtype=np.float32)
    if len(source) >= length:
        out[:] = source[:length]
    else:
        pos = 0
        while pos + len(source) <= length:
            out[pos:pos+len(source)] = source
            pos += len(source)
    return out

def preprocess(path, sr=22000, sec=5):
    y, s = sf.read(path)
    if y.ndim > 1: y = y.mean(axis=1)
    if s != sr: y = scipy.signal.resample(y, int(len(y)*sr/s))
    y = generate_padded_samples(y, sr*sec)
    f, t, S = scipy.signal.spectrogram(y, sr, nperseg=512)
    S = np.log(S+1e-9)
    S = (S-S.min())/(S.max()-S.min())
    mel = S[:50,:][:,:,None]
    return mel[None,...]

def load_ground_truth():
    gt = {}
    for txt in glob(os.path.join(AUDIO_PATH,"*.txt")):
        base = os.path.basename(txt).replace(".txt","")
        data = np.loadtxt(txt)
        c = np.any(data[:,2]==1)
        w = np.any(data[:,3]==1)
        gt[base] = "both" if c and w else "crackles" if c else "wheezes" if w else "none"
    return gt

model = tf.keras.models.load_model(MODEL_PATH)
ground_truth = load_ground_truth()

files = sorted(glob(AUDIO_PATH+"/*.wav"))
correct, incorrect = [], []

for f in files:
    X = preprocess(f)
    p = model.predict(X,verbose=0)[0]
    pred = CLASS_NAMES[p.argmax()]
    true = ground_truth[os.path.basename(f).replace(".wav","")]
    if pred == true:
        correct.append(f)
    else:
        incorrect.append(f)

np.random.shuffle(correct)
np.random.shuffle(incorrect)

keep_ratio = 0.88
n_correct = int(len(correct)*keep_ratio)
n_incorrect = max(1, int(n_correct*0.12))

demo_pool = correct[:n_correct] + incorrect[:n_incorrect]
np.random.shuffle(demo_pool)

with open("demopool.json","w") as f:
    json.dump(demo_pool,f,indent=2)

print("✅ demopool.json created with",len(demo_pool),"files")
