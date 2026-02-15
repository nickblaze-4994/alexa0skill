#WAV2.py
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import simpleaudio as sa
import torch
import torchaudio
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report,
                             roc_curve, auc,
                             precision_recall_curve)
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.utils import plot_model


# -----------------------------
# Data augmentation
# -----------------------------
def augment_audio(y, sr):
    """
    Return 5 extra variants covering:
    1. Pitch shift ±2 semitones
    2. Background noise mix  (≈10 dB SNR)
    3. Far-field / quiet + slight echo
    4. Low-frequency rumble  (flight turbulence)
    """
    aug = []

    # Convert input to tensor for pitch shifting
    y_tensor = torch.tensor(y).float()
    if len(y_tensor.shape) == 1:
        y_tensor = y_tensor.unsqueeze(0)  # Add batch dimension

    # 1-a Pitch ↑ 2 st
    aug.append(torchaudio.functional.pitch_shift(
        waveform=y_tensor,
        sample_rate=sr,
        n_steps=2
    ).squeeze().numpy())

    # 1-b Pitch ↓ 2 st
    aug.append(torchaudio.functional.pitch_shift(
        waveform=y_tensor,
        sample_rate=sr,
        n_steps=-2
    ).squeeze().numpy())

    # 2   Background-noise mix
    if len(_noise_clips) > 0:  # skip if no noise files
        # Randomly select index instead of using np.random.choice
        noise_idx = np.random.randint(0, len(_noise_clips))
        noise = _noise_clips[noise_idx]
        
        # Ensure noise length matches target length
        if len(noise) < len(y):
            noise = np.tile(noise, int(np.ceil(len(y)/len(noise))))[:len(y)]
        else:
            noise = noise[:len(y)]
            
        # Apply SNR mixing
        snr = 10  # dB
        rms_y = np.sqrt(np.mean(y**2))
        rms_n = np.sqrt(np.mean(noise**2))
        noise_scaled = noise * (rms_y / (10**(snr/20)) / (rms_n + 1e-6))
        aug.append(np.clip(y + noise_scaled, -1.0, 1.0))

    # 3   Far-field (-9 dB) + 40 ms echo
    quiet = y * 0.35
    echo = np.pad(quiet * 0.3, (int(0.04*sr), 0))[:len(y)]
    aug.append(np.clip(quiet + echo, -1.0, 1.0))

    # 4   Low-frequency rumble
    rumble = np.random.randn(len(y)) * 0.004
    # Apply low-pass filter
    b, a = scipy.signal.butter(4, 100/(sr/2), 'low')
    rumble = scipy.signal.filtfilt(b, a, rumble)
    aug.append(np.clip(y + rumble, -1.0, 1.0))

    return aug

# -----------------------------
# Noise-library loader
# -----------------------------
def load_noise_library(noise_dir):
    """Load noise files from directory"""
    noise_clips = []
    if os.path.exists(noise_dir):
        for fn in os.listdir(noise_dir):
            if fn.lower().endswith(".wav"):
                try:
                    y, _ = librosa.load(os.path.join(noise_dir, fn), sr=22050)
                    # Ensure the noise clip is 1D
                    if len(y.shape) > 1:
                        y = y.mean(axis=1)  # Convert stereo to mono
                    noise_clips.append(y)
                except Exception as e:
                    print(f"Error loading noise file {fn}: {e}")
    return noise_clips

# Initialize noise library
NOISE_DIR = "/Users/vijaysridhar/Documents/white noise"
_noise_clips = load_noise_library(NOISE_DIR)


# -----------------------------
# Audio Processing Functions
# -----------------------------
def load_and_process_audio(file_path, sr=22050, duration=None):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    return y, sr

def create_melspectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                               n_fft=n_fft, hop_length=hop_length)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect_db

def pad_or_truncate(mel_spect, target_length):
    if mel_spect.shape[0] > target_length: 
        return mel_spect[:target_length, :]
    else:
        pad_width = target_length - mel_spect.shape[0]
        return np.pad(mel_spect, ((0, pad_width), (0, 0)), mode='constant')

# -----------------------------
# Dataset Processing with Optional Augmentation
# -----------------------------
def process_audio_dataset(data_folder, classes, sr=22050, duration=3.0, n_mels=128, augment=False):
    features = []
    labels = []
    # Expected number of time frames (based on default hop_length=512)
    target_length = int(duration * sr / 512) + 1

    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(data_folder, class_name)
        if not os.path.isdir(class_path):
            print(f"Warning: Folder {class_path} not found.")
            continue
        for file in os.listdir(class_path):
            if file.endswith('.wav'):
                file_path = os.path.join(class_path, file)
                y, sr_ret = load_and_process_audio(file_path, sr=sr, duration=duration)
                if y is None:
                    continue
                # Original sample
                mel_spec = create_melspectrogram(y, sr_ret, n_mels=n_mels)
                mel_spec = mel_spec.T  # (time_steps, n_mels)
                mel_spec = pad_or_truncate(mel_spec, target_length)
                features.append(mel_spec)
                labels.append(class_index)
                # Append augmented samples if enabled
                if augment:
                    for y_aug in augment_audio(y, sr_ret):
                        mel_spec_aug = create_melspectrogram(y_aug, sr_ret, n_mels=n_mels)
                        mel_spec_aug = mel_spec_aug.T
                        mel_spec_aug = pad_or_truncate(mel_spec_aug, target_length)
                        features.append(mel_spec_aug)
                        labels.append(class_index)
    X = np.array(features)
    y = np.array(labels)
    return X, y

def prepare_data(X, y, test_split=0.2, val_split=0.2):
    from sklearn.model_selection import train_test_split
    # First split into training and temporary test set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_split, 
                                                        random_state=42, stratify=y)
    # Further split training set for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split, 
                                                      random_state=42, stratify=y_train)
    num_classes = len(np.unique(y))
    y_train = to_categorical(y_train, num_classes)
    y_val   = to_categorical(y_val, num_classes)
    y_temp  = to_categorical(y_temp, num_classes)
    # Add a channel dimension for CNN input
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_temp  = X_temp[..., np.newaxis]
    return (X_train, y_train), (X_val, y_val), (X_temp, y_temp)

# -----------------------------
# Model Building with Regularization
# -----------------------------
def build_model(input_shape, num_classes):
    reg = tf.keras.regularizers.l2(1e-4)
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=reg,
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=reg),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.optimizer.learning_rate.numpy())
    return model

# -----------------------------
# Model Training with Callbacks
# -----------------------------
def train_model(data_folder, classes, model_path, sr=22050, duration=3.0,
                n_mels=128, batch_size=32, epochs=30, augment=True):
    print("Processing dataset...")
    X, y = process_audio_dataset(data_folder, classes, sr, duration, n_mels, augment=augment)
    if len(X) == 0:
        print("No audio files processed. Check dataset path and file formats.")
        return
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(X, y)
    input_shape = X_train.shape[1:]  # (time_steps, n_mels, 1)
    num_classes = y_train.shape[1]
    
    print("Building and training the model...")
    model = build_model(input_shape, num_classes)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_val, y_val), callbacks=callbacks)
    
    loss, acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", acc)
    model.save(model_path)
    print("Model saved to", model_path)
    return model

# -----------------------------
# Inference and Chime-Playing Functions
# -----------------------------
def infer_trigger_word(model, file_path, sr=22050, duration=3.0, n_mels=128):
    y, sr_ret = load_and_process_audio(file_path, sr=sr, duration=duration)
    if y is None:
        print("Error: Could not load audio for inference.")
        return False
    mel_spec = create_melspectrogram(y, sr_ret, n_mels=n_mels)
    mel_spec = mel_spec.T
    target_length = int(duration * sr / 512) + 1
    mel_spec = pad_or_truncate(mel_spec, target_length)
    X_in = np.expand_dims(mel_spec, axis=0)      # (1, time_steps, n_mels)
    X_in = np.expand_dims(X_in, axis=-1)         # (1, time_steps, n_mels, 1)
    pred = model.predict(X_in)
    # Assuming class index 1 corresponds to the trigger word (e.g., "activate")
    trigger_prob = pred[0][1]
    print("Trigger word probability:", trigger_prob)
    return trigger_prob >= 0.5  # Adjust threshold if needed

def play_chime(chime_file):
    try:
        wave_obj = sa.WaveObject.from_wave_file(chime_file)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print("Error playing chime:", e)


# -----------------------------
# Sliding‑window inference
# -----------------------------
def sliding_window_infer(model, file_path, sr=22050,
                         win_dur=3.0, hop_dur=0.5,
                         n_mels=128, thresh=0.25):
    y, sr = librosa.load(file_path, sr=sr)
    win_len = int(win_dur * sr)
    hop_len = int(hop_dur * sr)
    target_len = int(win_dur * sr / 512) + 1

    for start in range(0, len(y) - win_len + 1, hop_len):
        chunk = y[start:start + win_len]
        mel = create_melspectrogram(chunk, sr, n_mels).T
        mel = pad_or_truncate(mel, target_len)
        X = mel[np.newaxis, ..., np.newaxis]
        prob = model.predict(X, verbose=0)[0][1]
        print(f"{start/sr:5.1f}s → prob={prob:.3f}")  # Optional debug print
        if prob >= thresh:
            print(f"✅ Trigger word detected at ≈ {start/sr:.1f}s (prob={prob:.2f})")
            return True
    print("Trigger word not detected.")
    return False



# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Choose mode: set 'mode' to either 'train' or 'infer'
    mode = 'train'  # Change to 'infer' for detection testing

    # Common parameters
    SAMPLE_RATE = 22050
    DURATION = 3.0
    N_MELS = 128
    MODEL_PATH = "/Users/vijaysridhar/Documents/trigger_model.keras"
    
    if mode == 'train':
        # Path to your dataset folder (with subfolders for each class)
        DATA_FOLDER = "/Users/vijaysridhar/Documents/WAV"  # e.g., folders "negative" and "activate"
        CLASSES = ["negative", "activate"]  # "activate" is the trigger word class
        # Set augment=True to enable data augmentation
        train_model(DATA_FOLDER, CLASSES, MODEL_PATH, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS, augment=True)
    elif mode == 'infer':
        AUDIO_FILE = "/Users/vijaysridhar/Documents/inference/clip_5.wav"
        CHIME_FILE = "/Users/vijaysridhar/Documents/inference/chime.wav"
        model = load_model(MODEL_PATH)
        detected = sliding_window_infer(
            model,
            AUDIO_FILE,
            sr=SAMPLE_RATE,
            win_dur=3.0,
            hop_dur=0.5,
            n_mels=N_MELS,
            thresh=0.25)
        if detected:
            print("Trigger word detected! Playing chime.")
            play_chime(CHIME_FILE)
        else:
            print("Trigger word not detected.")

