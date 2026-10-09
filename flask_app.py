from flask import Flask, request, render_template
import sklearn
import numpy as np
import pandas as pd
import joblib


obj=joblib.load('california.joblib')
model=obj['Model']
columns=obj['Columns']


app=Flask(__name__)
@app.route('/')
def main():
    return('welcome')


@app.route('/predict')
def predict():
    Input=[]
    for i in columns:
        val=request.args.get(i,type=float)
        Input.append(val)
    out=model.predict([Input])
    return(out)


if __name__=="__main__":
    app.run(debug=True)
