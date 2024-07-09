#Server code for running machine learning code
import dill

import pandas as pd 

from fastapi import FastAPI
from pydantic import BaseModel 

from sklearn.preprocessing import StandardScaler

app = FastAPI()
pipe = dill.load(open('./models/flight_cost.pkl', mode = 'rb'))

#standart scaler of the target variable
df = pd.read_csv(filepath_or_buffer = './data/train_data.csv', sep = ",")
y = df[['price']]
scale_up = StandardScaler()
scale_up.fit(X = y)

class Form(BaseModel):
    id : int
    airline : str
    flight :  str
    source_city : str
    departure_time : str 
    stops : str
    arrival_time : str 
    destination_city : str 
    travel_class : str 
    duration : float
    days_left : int

class Prediction(BaseModel):
    Result : float

@app.get('/status')
def status():
    return "I'm OK"

@app.get('/version')
def version():
    return pipe['metadata']

@app.post('/predict', response_model = Prediction)
def predict(form : Form):
    df = pd.DataFrame.from_dict([form.dict()])
    df.rename(columns={'travel_class' : 'class'}, inplace=True)
    y = pipe['model'].predict(df)
    return {
        'Result' : scale_up.inverse_transform(y[0].reshape(-1, 1)),
    }