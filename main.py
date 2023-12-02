import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = './Deep4SNet/model_Deep4SNet.h5'
weights_path = './Deep4SNet/weights_Deep4SNet.h5'
model = load_model(model_path)
model.load_weights(weights_path)


def process_audio(file_path):
    filename = os.path.basename(file_path)

    y, sr = librosa.load(file_path, sr=None)

    short_time_fourier_transform = librosa.stft(y)
    magnitude = np.abs(short_time_fourier_transform)
    mel_spectogram = librosa.feature.melspectrogram(S=magnitude, sr=sr, n_mels=128)
    mel_spectogram_decibel_scale = librosa.power_to_db(mel_spectogram, ref=np.max)

    plt.figure(figsize=(1.5, 1.5))
    librosa.display.specshow(mel_spectogram_decibel_scale, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(f'generated-spectograms/{filename}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    img = cv2.imread(f'generated-spectograms/{filename}.png')
    img = cv2.resize(img, (150, 150))
    img = np.reshape(img, [1, 150, 150, 3])

    return img


def predict_deepfake(file_path):
    processed_audio = process_audio(file_path)
    prediction = model.predict(processed_audio, batch_size=19)

    return prediction[0][0]


if __name__ == '__main__':
    if len(sys.argv) == 1:
        file_path = sys.argv[1]
    else:
        print("Please provide a path to the audio file.")
        sys.exit(-1)

    result = predict_deepfake(file_path)

    if result > 0.5:
        print(f"The voice recording '{file_path}' is original.")
    else:
        print(f"The voice recording '{file_path}' is fake.")
