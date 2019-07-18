import numpy as np
import flask
#import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list):
    #data = pd.DataFrame(to_predict_list)
    #data = data[[4, 6, 5, 3, 2, 0, 1]]
    to_predict = np.asarray(to_predict_list)
    #myorder = [3, 5, 4, 2, 0, 1]
    #myorder = [5, 1, 0, 3, 2, 4]
    #to_predict = [to_predict[i] for i in myorder]
    loaded_model = pickle.load(open("model.pkl","rb"), encoding = 'latin1')
    result = loaded_model.predict([list(map(float,to_predict))])
    return result[0][0]*100
    #return to_predict


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        result = ValuePredictor(to_predict_list)
            
        return render_template("result.html",prediction=result)

if __name__ == '__main__':
    app.run(debug=True)