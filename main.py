from flask import Flask, render_template, request
import pickle
import json
import numpy as np
import CONFIG

with open(CONFIG.MODEL_PATH,'rb') as file:
    model = pickle.load(file)

with open(CONFIG.ASSET_PATH,'r') as file:
    asset = json.load(file)
col = asset['columns']
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/get_data", methods = ["GET","POST"])
def data():
    input_data = request.form
    print(input_data)
    
    data = np.zeros(len(col)+1)
    data[0] = input_data['age']
    data[1] = input_data['sex']
    data[2] = input_data['cp']
    data[3] = input_data['trtbps']
    data[4] = input_data['chol']
    data[5] = input_data['fbs']
    data[6] = input_data['restecg']
    data[7] = input_data['thalachh']
    data[8] = input_data['exng']
    data[9] = input_data['oldpeak']
    data[10] = input_data['slp']
    data[11] = input_data['caa']
    data[12] = input_data['thall']
    

    
    result = model.predict([data])
    print(result)

    if result[0] == 0:
        Heart_attack = "No"
    if result[0] == 1:
        Heart_attack = "Yes"

    return render_template("index.html",PREDICT_VALUE=Heart_attack)

if __name__ == "__main__":
    print(len(col))
    app.run(host=CONFIG.HOST_NAME, port= CONFIG.PORT_NUMBER)