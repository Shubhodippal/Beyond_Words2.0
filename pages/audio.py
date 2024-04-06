import streamlit as st
import sounddevice as sd
import soundfile as sf
import librosa
import soundfile
import pickle
import numpy as np

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

def record_audio(duration, filename):
    # Record audio from the default microphone
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()

    # Save the recorded audio to a file
    sf.write(filename, audio, samplerate)

# Set up Streamlit UI
st.title("Audio Recorder")

duration = st.number_input("Recording duration (seconds):", min_value=1, max_value=60, value=5, step=1)
filename = st.text_input("Enter filename to save:", "recorded_audio.wav")
samplerate = 48100  # Sample rate in Hz

if st.button("Start Recording"):
    st.write("Recording...")
    record_audio(duration, filename)
    st.write(f"Recording saved as {filename}")

model = 'modelForPrediction1.sav'
loaded_model = pickle.load(open(model, 'rb')) # loading the model file from the storage

if (filename):
    feature=extract_feature(filename, mfcc=True, chroma=True, mel=True)

    feature=feature.reshape(1,-1)

    prediction=loaded_model.predict(feature)
    st.write(prediction)