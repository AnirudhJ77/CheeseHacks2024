import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import librosa as lr
from pathlib import Path

def extract_features(audio_path):
    audio_data, sample_rate = lr.load(audio_path, sr=None)

    # Define frame duration (in seconds) and calculate frame_length and hop_length
    frame_duration_sec = 0.05  # 50ms frame duration
    frame_length = int(frame_duration_sec * sample_rate)
    hop_length = int(frame_length * 0.5)  # 50% overlap

    # Extract features with lr
    features = [
        lr.feature.zero_crossing_rate(y=audio_data, frame_length=frame_length, hop_length=hop_length), #zero_crossing_rate
        lr.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length), #rms
        lr.feature.spectral_centroid(y=audio_data, sr=sample_rate, hop_length=hop_length), #spectral_centroid
        lr.feature.spectral_bandwidth(y=audio_data, sr=sample_rate, hop_length=hop_length), #spectral_bandwidth
        lr.feature.spectral_rolloff(y=audio_data, sr=sample_rate, hop_length=hop_length),#spectral_rolloff
        lr.feature.spectral_flatness(y=audio_data, hop_length=hop_length), #spectral_flatness
        lr.feature.mfcc(y=audio_data, sr=sample_rate, hop_length=hop_length, n_mfcc=13),#mfcc
        lr.feature.chroma_stft(y=audio_data, sr=sample_rate, hop_length=hop_length),#chroma_stft
        lr.feature.tonnetz(y=lr.effects.harmonic(audio_data), sr=sample_rate)#tonnetz
    ]

    # Calculate the mean value for each feature across time (over all frames)
    mean_features = []

    for feature_array in features:
        if feature_array.ndim == 1:
            # For 1D features (e.g., zero-crossing rate, rms), calculate mean over the frames
            mean_features.append(np.mean(feature_array))
        else:
            # For 2D features (e.g., MFCC, chroma), calculate the mean over the frames (axis=1)
            mean_features.extend(np.mean(feature_array, axis=1))

    # Flatten all the mean features into a single 1D array
    final_features = np.array(mean_features)
    return final_features

def normalize_features(data):
    range_features = np.max(a=data, axis=1) - np.min(a=data, axis=1)
    norm_data = data/range_features
    return norm_data

def standardize_features(data):
    mean_features = np.mean(a=data, axis=1)
    std_features = np.std(a=data, axis=1)
    std_data = (data-mean_features)/std_features
    return std_data

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

data_folder = Path("dataset/")
nrows = 440
ncols = 37
def load_data(type='emotion'):
    x = []  # Use a list to dynamically store features
    y = []  # Use a list to store corresponding labels

    for file in data_folder.iterdir():
        if not file.is_file() or file.suffix != '.wav': 
            continue
        try:
            # Extract features and label
            features = extract_features(str(file))
            label = get_y(file.name, type)
            # Append to lists
            x.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    # Convert lists to arrays
    data = np.array(x)
    labels = np.array(y)

    # Save to CSV
    pd.DataFrame(data).to_csv('data.csv', index=False)
    pd.DataFrame(labels).to_csv('labels.csv', index=False)

    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    return data, labels

num_emotions = 3
load_data()
# model = models.Sequential([
#     layers.Input(shape=(X_train.shape[1],)),  # Input: num_features
#     layers.BatchNormalization(),
#     layers.Dense(128, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.3),
#     layers.Dense(64, activation='relu', kernel_regularizer='l2'),
#     layers.Dropout(0.2),
#     layers.Dense(num_emotions, activation='softmax')  # Output: num_emotions
# ])

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# model.summary()

# def predict(audio):
#     pass