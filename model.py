import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import librosa as lr
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def extract_features(audio_path):
    audio_data, sample_rate = lr.load(audio_path, sr=16000)  # Resample to 16 kHz

    # Define frame duration (in seconds) and calculate frame_length and hop_length
    frame_duration_sec = 0.05  # 50ms frame duration
    frame_length = int(frame_duration_sec * sample_rate)
    hop_length = int(frame_length * 0.5)  # 50% overlap

    # Extract features with lr, ensuring fmax is within Nyquist frequency
    fmax = sample_rate / 2
    n_fft = 256
    features = [
        lr.feature.zero_crossing_rate(y=audio_data, frame_length=frame_length, hop_length=hop_length),  # zero_crossing_rate
        lr.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length),  # rms
        lr.feature.spectral_centroid(y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=n_fft//4),  # spectral_centroid
        lr.feature.spectral_bandwidth(y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=n_fft//4),  # spectral_bandwidth
        lr.feature.spectral_rolloff(y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=n_fft//4),  # spectral_rolloff
        lr.feature.spectral_flatness(y=audio_data, n_fft=n_fft, hop_length=n_fft//4),  # spectral_flatness
        lr.feature.mfcc(y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=n_fft//4, n_mfcc=13),  # mfcc
        lr.feature.chroma_stft(y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=n_fft//4, n_chroma=12),  # chroma_stft
        lr.feature.tonnetz(y=lr.effects.harmonic(audio_data), sr=sample_rate),  # tonnetz
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
    range_features = np.max(data, axis=0) - np.min(data, axis=0)
    norm_data = data/range_features
    return norm_data

def standardize_features(data):
    mean_features = np.mean(a=data, axis=0)
    std_features = np.std(a=data, axis=0)
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
            features = extract_features(file)
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

x = pd.read_csv("data.csv").to_numpy()
y = pd.read_csv('labels.csv').to_numpy()

x_norm = normalize_features(x)

model = models.Sequential([
    layers.Input(shape=(x_norm.shape[1],)),  # Input: num_features
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

history = model.fit(x_norm, y, epochs=44, batch_size=32, validation_split=0.2)

model.summary()

model.save('cat_emotion_model_first_try.keras')

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

def predict(audio):
    new_X = extract_features(audio)
    model = load_model('cat_emotion_model_first_try.keras')
    predictions = model.predict(new_X)  # new_X: feature array of a new sample
    predicted_class = predictions.argmax(axis=-1)
    idx = np.argmax(predicted_class)
    return emotion_classes[idx]
print(predict('meow.wav'))