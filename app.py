
# coding: utf-8

import pandas as pd
from adultIncomeClassifier import logger
from adultIncomeClassifier.components.model_trainer import IncomeClassifierModel 
from adultIncomeClassifier.utils import load_object
from pathlib import Path
from flask import Flask, request, render_template
import os

model_pkl_file_path = "artifacts/model_trainer/model.pkl"

app = Flask("__name__",template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('home.html')



@app.route("/predict", methods=['POST'])
def predict():
    
    '''
    age: int64
    sex: object
    fnlwgt: int64
    education: object
    marital-status: object
    occupation: object
    relationship: object
    race: object
    workclass: object
    capital-gain: int64
    capital-loss: int64
    hours-per-week: int64
    country: object
    '''
    

    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']


    model =load_object(file_path=Path(model_pkl_file_path))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13,'12','0']]
    
    df = pd.DataFrame(data, columns = ['age', 'sex', 'fnlwgt', 'education', 
                                           'marital-status', 'occupation', 'relationship', 
                                           'race', 'workclass', 'capital-gain', 
                                           'capital-loss', 'hours-per-week', 'country',
                                           'education-num','salary'
                                    ])
    
 
    for col in ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']:
        df[col] = df[col].astype('int64')

    #logger.info(df.values)
    output = model.predict(df)

    
    if output[0] == 1:
        o1 = "This person income falls in more than 50k (>50k) side !"
    else:
        o1 = "This person income falls in less than 50k (<=50k) side !"
    
    o2 = "Thanks for using Income Classification Model."
        
    return render_template('home.html', output1=o1, output2=o2, 
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'], 
                           query7 = request.form['query7'], 
                           query8 = request.form['query8'], 
                           query9 = request.form['query9'], 
                           query10 = request.form['query10'], 
                           query11 = request.form['query11'], 
                           query12 = request.form['query12'], 
                           query13 = request.form['query13']
                        )


if __name__=="__main__":
    app.run(debug=True)