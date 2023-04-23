from flask import Flask, render_template, request
import pandas as pd 
import numpy as np
from pyabsa import ATEPCCheckpointManager
from keras.utils import np_utils
import librosa, resampy
from keras.models import model_from_json

app = Flask(__name__)

# Load audio model
json_file = open('audio-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("Emotion_Voice_Detection_Model14TH.h5")

# Load text model
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english')

@app.route('/home')
def home():
    return "hello!!!!! <3"
    # return render_template('home.html')

@app.route("/text", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # feedback = requests.form['feedback']
            feedback = ['Very redundant course, it\'s just data structures but harder. If you add a little more content to data structures, this course would be entirely redundant',]

            atepc_result = aspect_extractor.extract_aspect(inference_source=feedback, pred_sentiment=True,)
            dic = {'aspect': atepc_result[0]['aspect'],
                   'sentiment': atepc_result[0]['sentiment'],
                   'confidence': atepc_result[0]['confidence'],
            }
            atepc_result_df = pd.DataFrame(dic)

        except valueError:
            return "Please check if values are entered correctly!!!!!!!! :("

    # return render_template('text.html', prediction = atepc_result_df)
    return "text page here :)"

@app.route("/audio", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            X, sample_rate = librosa.load('audiorec4.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
            featurelive = mfccs
            livedf2 = featurelive
  
            livedf2= pd.DataFrame(data=livedf2)
            livedf2 = livedf2.stack().to_frame().T
            twodim= np.expand_dims(livedf2, axis=2)

            livepreds = loaded_model.predict(twodim, 
                                    batch_size=32, 
                                    verbose=1)

            livepreds1=livepreds.argmax(axis=1)

            liveabc = livepreds1.astype(int).flatten()       

            if liveabc[0] == 0:
                predicted_voice= "female_happy"
            elif liveabc[0] == 1:
                predicted_voice="female_neutral"
            elif liveabc[0] == 2:
                predicted_voice="female_sad"
            elif liveabc[0] == 3:
                predicted_voice="male_happy"
            elif liveabc[0] == 4:
                predicted_voice="male_neutral"
            elif liveabc[0] == 5:
                predicted_voice="male_sad "

    
        except valueError:
            return "Please check if values are entered correctly!!!!!!!! :("

    # return render_template('audio.html', prediction = predicted_voice)
    return "audio page"
    
if __name__ == "__main__":
    app.run()