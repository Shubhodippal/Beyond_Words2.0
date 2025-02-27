import librosa
import soundfile
import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score
import pyaudio
import wave

i=0
acc=[]

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to observe
observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                      hidden_layer_sizes=(300), learning_rate='adaptive', max_iter=500)


# Extract features (mfcc, chroma, mel) from a sound file
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

# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("D:\Coding\Project\Beyond_Words2.0\Beyond_Words2.0\speech-emotion-recognition-ravdess-data\*\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        
        #print("File name = {} , emotion = {}".format(file_name, emotion))
        
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# to train model if started program
def trainModel():

    # Split the dataset
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)
    print((x_train.shape[0], x_test.shape[0]))

    # Get the number of features extracted
    print(f'Features extracted: {x_train.shape[1]}')

    # Train the model
    model.fit(x_train, y_train)

    # Predict for the test set
    y_pred = model.predict(x_test)

    # Calculate the accuracy of our model
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    # Print the accuracy
    print("Accuracy: {:.2f}%".format(accuracy*100))

    f1=f1_score(y_test, y_pred,average='weighted')
    print(f1)
    with open('ser_models\ser_f1.txt', 'a') as file:
        file.write(f"{f1}\n")

    df=pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
    df.head(20)

    
    return (accuracy*100)



while True:
    print(f"in {i} th iteraion")
    accr = trainModel()
    acc.append(accr)
    with open('ser_models\ser_accr.txt', 'a') as file:
        file.write(f"{accr}\n")
    with open( f"ser_models\{i}th_modelForPrediction_{accr}%.sav", 'wb') as f:
        pickle.dump(model,f)
    print("\n-------------------------------------------------------------------------------------------------------\n")
    i+=1