from fastapi import FastAPI
from pydantic import BaseModel
# import h5py
import tensorflow as tf
filename = "Diabetes_model.h5"
# import pickle 
import pandas as pd
import numpy as np

app = FastAPI()

class ScoringResult(BaseModel):
    # user_data = {"Glucose": 79,
    #     "Systolic Blood Pressure": 118,
    #     "Diastolic Blood Pressure": 73,
    #     "BMI": 98,
    #     "Age": 9}
    Glucose : int
    Sys_BP : int
    Dia_BP :  int
    BMI : int
    Age : int

model = tf.keras.models.load_model('Diabetes_model.h5')        
# with h5py.File('DiaCare.h5', "r") as hi:
#     model = pickle.load(hi)

@app.post("/predict")
async def root(result: ScoringResult):
    # input_df: pd.DataFrame = pd.DataFrame([user_data])
    
    df = pd.DataFrame([result.dict().values()], columns=result.dict().keys())
    result = model.predict(df)
    if int(result):
        return "The patient is Diabetic"
    else:
        return "The patient is not Diabetic" 
    # return {"prediction":int(result)}



    # data = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    # pred = model.predict(data)
    # return {"prediction":int(pred)}



