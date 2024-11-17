import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa as lr
from pathlib import Path

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
    final_features = np.hstack(list(mean_features.values()))

    print("Final Feature Array Shape:", final_features.shape)
    print("Final Feature Array:", final_features)


def normalize_features(data):
    """
    Takes in a dataset and normalizes each feature to be between 0 and 1.
    x_norm = x/(max(x)-min(x))
    Parameters:
    data (np.array) : Dataset of Features
    Returns:
    (np.array) : Dataset of normalized features
    """
def standardize_features(data):
    """
    Takes in a dataset and normalizes each feature to have mean 0 and variance 1
    x_norm = (x-mean(x))/sd(x)
    Parameters:
    data (np.array) : Dataset of Features
    Returns:
    (np.array) : Dataset of normalized features
    """
emotion_classes = {
    0 : "Satisfied",
    1 : "Hungry",
    2 : "Anxious"
}
def get_y(filename, type='emotion'):
    """
    Encodes the output labels using one-hot encoding
    """
    if type=='emotion' :
        y = filename[0]
        if y == 'B':
            return np.array([1, 0, 0])
        elif y == 'F':
            return np.array([0, 1, 0])
        elif y=='I':
            return np.array([0, 0, 1])
    return None

data_folder = Path("./dataset/")
nrows = 440
ncols = 37
def load_data(type='emotion'):
    data = np.empty(shape=(nrows, ncols))
    y = np.empty(shape=(nrows, num_emotions))
    count = 0
    for file in data_folder.iterdir():
        if not file.is_file() or file.suffix != '.wav': 
            continue
        filename = file.name
        data[count] = extract_features(filename)
        y[count] = get_y(filename, type)
        count+=1
    pd.DataFrame(data).to_csv('data.csv')
    pd.DataFrame(y).to_csv('labels.csv')

num_emotions = 3
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # Input: num_features
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.2),
    layers.Dense(num_emotions, activation='softmax')  # Output: num_emotions
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

def predict(audio):
    pass