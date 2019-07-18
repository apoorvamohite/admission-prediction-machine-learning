import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.asarray(to_predict_list)
    loaded_model = pickle.load(open("model.pkl","rb"), encoding = 'latin1')
    result = loaded_model.predict([list(map(float,to_predict))])
    return result[0][0]*100

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        result = ValuePredictor(to_predict_list)
        return render_template("result.html",prediction=result)

if __name__ == '__main__':
    app.run(debug=True)