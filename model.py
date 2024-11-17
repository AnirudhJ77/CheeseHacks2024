import numpy as np
import pandas as pd
import librosa as lr

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

print(extract_features("meow.wav"))