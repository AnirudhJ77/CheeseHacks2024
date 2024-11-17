import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa as lr
def load_data():
    pass

def extract_features(audio_path):
    audio_data, sample_rate = lr.load(audio_path, sr=None)

    # Define frame duration (in seconds) and calculate frame_length and hop_length
    frame_duration_sec = 0.05  # 50ms frame duration
    frame_length = int(frame_duration_sec * sample_rate)
    hop_length = int(frame_length * 0.5)  # 50% overlap

    # Extract features with lr
    features = {
        "zero_crossing_rate": lr.feature.zero_crossing_rate(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0],
        "rms": lr.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0],
        "spectral_centroid": lr.feature.spectral_centroid(y=audio_data, sr=sample_rate, hop_length=hop_length)[0],
        "spectral_bandwidth": lr.feature.spectral_bandwidth(y=audio_data, sr=sample_rate, hop_length=hop_length)[0],
        "spectral_rolloff": lr.feature.spectral_rolloff(y=audio_data, sr=sample_rate, hop_length=hop_length)[0],
        "spectral_flatness": lr.feature.spectral_flatness(y=audio_data, hop_length=hop_length)[0],
        "mfcc": lr.feature.mfcc(y=audio_data, sr=sample_rate, hop_length=hop_length, n_mfcc=13),
        "chroma_stft": lr.feature.chroma_stft(y=audio_data, sr=sample_rate, hop_length=hop_length),
        "tonnetz": lr.feature.tonnetz(y=lr.effects.harmonic(audio_data), sr=sample_rate),
    }

    # Calculate the mean value for each feature across time (over all frames)
    mean_features = {}

    for key, feature_array in features.items():
        if feature_array.ndim == 1:
            # For 1D features (e.g., zero-crossing rate, rms), calculate mean over the frames
            mean_features[key] = np.mean(feature_array)
        else:
            # For 2D features (e.g., MFCC, chroma), calculate the mean over the frames (axis=1)
            mean_features[key] = np.mean(feature_array, axis=1)

    # Flatten all the mean features into a single 1D array
    final_features = np.array(list(mean_features.values()))

    print("Final Feature Array Shape:", final_features.shape)
    print("Final Feature Array:", final_features)
    
    return final_features

print(extract_features("meow.wav"))

def normalize_features(data):
    cols = len(data[0])

    data_norm = []

    for col in range(cols):
        feature = [row[col] for row in data]

        feature_norm = []

        range = max(feature)-min(feature)

        for val in feature:
            feature_norm.append(val/range)

        data_norm.append(feature_norm)
    
    return np.array(data_norm)

def standardize_features(data):
    cols = len(data[0])

    data_std = []

    for col in range(cols):
        feature = [row[col] for row in data]

        feature_std = []

        mean = np.mean(feature)
        stddev = np.std(feature)

        for val in feature:
            feature_std.append((val-mean)/stddev)

        data_std.append(feature_std)
    
    return np.array(data_std)

def encode_labels(y_train, y_test=None):
    """
    Encodes the output labels
    """
def test_train_split(data):
    """
    Splits the data into test and train data
    """

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # Input: num_features
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')  # Output: num_classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()