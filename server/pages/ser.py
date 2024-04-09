import streamlit as st
import sounddevice as sd
import soundfile as sf
import librosa
import soundfile
import pickle
import numpy as np
import os

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

def main():
    st.title("Beyond Words")
    
    record = st.checkbox('Record audio')
    upload = st.checkbox('Detect from audio file')

    if(record):
        st.write("This feature will be avilable soon")
    elif(upload):
        uploaded_file = st.file_uploader("Choose a .wav file", type="wav")
        
        if uploaded_file is not None:
            with open("audio.wav", "wb") as f:
                f.write(uploaded_file.getvalue())
            st.success("File saved as audio.wav")
            st.audio("audio.wav", format='audio/wav')
            filename = "audio.wav"

            model = '49th_modelForPrediction_57.22222222222222%.sav'
            loaded_model = pickle.load(open(model, 'rb')) # loading the model file from the storage

            if (os.path.exists(filename)):
                run = st.button("Analyze")
                feature=extract_feature(filename, mfcc=True, chroma=True, mel=True)

                feature=feature.reshape(1,-1)

                prediction=loaded_model.predict(feature)

                if run:
                    st.write(prediction)
                Del =  st.button('Delete Data')
                if Del:
                    if os.path.exists(filename) :
                        os.remove("audio.wav")
                    else:
                        st.write("The file has allready been deleted or it doesnt exist.")

if __name__ == "__main__":
    main()
                    