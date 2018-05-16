from flask import Flask,jsonify,request
from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
Swagger(app)
CORS(app)
@app.route('/')
def hello_world():
    return "Glass Classification"


@app.route('/input',methods=['POST'])
def inputData():
    """
    Ini adalah Endpoint untuk memprediksi Data Kaca
    ---
    tags :
        - Rest Controller
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: Glass
          required:
            -RefractingIndex
            -Natrium
            -Magnesium
            -Aluminium
            -Silicone
            -Potassium
            -Calcium
            -Barium
            -Iron
          properties:
            RefractingIndex:
              type: float
              description: Please input with valid input range between 1.511150 ~ 1.533930.
              default: 0
            Natrium:
              type: float
              description: Please input with valid input range between 10.730000 ~ 17.380000.
              default: 0
            Magnesium:
              type: float
              description: Please input with valid input range between 0 ~ 4.490000.
              default: 0
            Aluminium:
              type: float
              description: Please input with valid input range between 0.290000 ~ 3.500000.
              default: 0
            Silicone:
              type: float
              description: Please input with valid input range between 69.810000 ~ 75.410000.
              default: 0
            Potassium:
              type: float
              description: Please input with valid input range between 0 ~ 6.210000.
              default: 0
            Calcium:
              type: float
              description: Please input with valid input range between 5.430000 ~ 16.190000.
              default: 0
            Barium:
              type: float
              description: Please input with valid input range between 0 ~ 3.150000.
              default: 0
            Iron:
              type: float
              description: Please input with valid input range between 0 ~ 0.510000.
              default: 0
    response:
        200:
            description: Success Input
    """
    dataBaru=request.get_json()

    RefractingIndex = dataBaru['RI']
    Natrium = dataBaru['Na']
    Magnesium = dataBaru['Mg']
    Aluminium = dataBaru['Al']
    Silicone = dataBaru['Si']
    Potassium = dataBaru['K']
    Calcium = dataBaru['Ca']
    Barium = dataBaru['Ba']
    Iron = dataBaru['Fe']


    glassBaru=np.array([[RefractingIndex, Natrium, Magnesium, Aluminium, Silicone, Potassium, Calcium, Barium, Iron]])
    clf=joblib.load('static/randomForestClassifier.pkl')
    resultPredict=clf[0].predict(glassBaru)
    return jsonify({'message':format(resultPredict)})
