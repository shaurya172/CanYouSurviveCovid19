import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_lr.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = ['sex',
 'pneumonia',
 'age',
 'diabetes',
 'copd',
 'asthma',
 'inmsupr',
 'hypertension',
 'other_disease',
 'cardiovascular',
 'obesity',
 'renal_chronic',
 'tobacco',
 'contact_other_covid',
 'Test result'
 ]
    user_input = [int(x) for x in request.form.values()]
    df = pd.DataFrame([user_input],columns=features)
    prediction = model.predict(df)
    prediction = ''.join(prediction)

    output = prediction

    return render_template('index.html', prediction_text='Will Covid be Fatal for you: {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    
