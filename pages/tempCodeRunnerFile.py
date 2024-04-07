run=st.button("Analyze Audio")

if run:
    filename = "recorded_audio.wav"
    model = '49th_modelForPrediction_57.22222222222222%.sav'
    loaded_model = pickle.load(open(model, 'rb')) # loading the model file from the storage

    if (filename is not None):
        feature=extract_feature(filename, mfcc=True, chroma=True, mel=True)

        feature=feature.reshape(1,-1)

        prediction=loaded_model.predict(feature)
        st.write(prediction)
    Del =  st.button('Delete Data')
    if Del :
        os.remove("recorded_audio.wav")
        