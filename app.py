import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model_nc = pickle.load(open('model_nc.pkl', 'rb'))
model_uc = pickle.load(open('model_uc.pkl', 'rb'))
model_nb = pickle.load(open('model_nb.pkl', 'rb'))
model_ub = pickle.load(open('model_ub.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/newcar', methods=['POST', 'GET'])
def rnewcar():
    return render_template('resultnewcar.html')



@app.route('/newbike', methods=['POST', 'GET'])
def rnewbike():
    return render_template('resultnewbike.html')


@app.route('/usedbike', methods=['POST', 'GET'])
def rusedbike():
    return render_template('resultusedbike.html')



@app.route('/usedcar', methods=['POST', 'GET'])
def rusedcar():
    return render_template('resultusedcar.html')


@app.route('/resultnewcar.html', methods=['POST', 'GET'])
def newcar():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_nc.predict(final_features)
    prediction=np.exp(prediction)
    output = prediction[0]
    return render_template('resultnewcar.html', prediction_text='Resale value :{}'.format(output))


@app.route('/resultnewbike.html', methods=['POST', 'GET'])
def newbike():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_nb.predict(final_features)
    prediction=np.exp(prediction)
    output = prediction[0]
    return render_template('resultnewbike.html', prediction_text='Resale value :{}'.format(output))


@app.route('/resultusedbike.html', methods=['POST', 'GET'])
def usedbike():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_ub.predict(final_features)
    prediction=np.exp(prediction)
    output = prediction[0]
    return render_template('resultusedbike.html', prediction_text='Resale value :{}'.format(output))

@app.route('/resultusedcar.html', methods=['POST', 'GET'])
def usedcar():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_uc.predict(final_features)
    prediction=np.exp(prediction)
    output = prediction[0]
    return render_template('resultusedcar.html', prediction_text='Resale value :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)