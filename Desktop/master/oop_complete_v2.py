from fastapi import FastAPI, HTTPException
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from pydantic import BaseModel , Field
from typing import Optional



app = FastAPI()





class LoadData:
    def __init__(self, 
                 heart_train=r"C:\Users\grigr\Desktop\master\heart_train.csv",
                 heart_test=r"C:\Users\grigr\Desktop\master\heart_test.csv"):
        self.heart_train = pd.read_csv(heart_train) 
        self.heart_test = pd.read_csv(heart_test)
        
    def fill_na_lower(self):

        self.heart_train.columns = self.heart_train.columns.str.lower()
        self.heart_test.columns = self.heart_test.columns.str.lower()

        
        num_col = ['age',
'cholesterol', 'heart rate', 'exercise hours per week',
'sedentary hours per day', 'income', 'bmi',	'triglycerides', 'physical activity days per week','sleep hours per day','blood sugar',
'ck-mb' ,'troponin',
'systolic blood pressure',	'diastolic blood pressure', 'stress level'
]
        cat_col = [
            'diabetes','family history','smoking','obesity', 'alcohol consumption', 'previous heart problems',
'medication use', 'diet',  'gender'
        ]

        self.heart_train['gender'] = self.heart_train['gender'].apply(lambda x: 
    1 if x in ['Male', '1.0'] else 0
)

        self.heart_test['gender'] = self.heart_test['gender'].apply(lambda x: 
    1 if x in ['Male', '1.0'] else 0
)
        
        for column in cat_col:
            if column in self.heart_test.columns:
                if self.heart_test[column].isnull().mean() < 0.1:
                    mode_value_test = self.heart_test[column].mode()[0]
                    self.heart_test[column] = self.heart_test[column].fillna(mode_value_test)
                    
        for column in cat_col:
            if column in self.heart_train.columns:
                if self.heart_train[column].isnull().mean() < 0.1:
                    mode_value_test = self.heart_train[column].mode()[0]
                    self.heart_train[column] = self.heart_train[column].fillna(mode_value_test)

        for column in num_col:
            if column in self.heart_test.columns:
                if self.heart_test[column].isnull().mean() < 0.1:
                    median_value_test = self.heart_test[column].median()
                    self.heart_test[column] = self.heart_test[column].fillna(median_value_test)
                    
        for column in num_col:
            if column in self.heart_train.columns:
                if self.heart_train[column].isnull().mean() < 0.1:
                    median_value_test = self.heart_train[column].median()
                    self.heart_train[column] = self.heart_train[column].fillna(median_value_test)

class split_heart(LoadData):
    def __init__(self):
        super().__init__()
        self.fill_na_lower()

    def split(self):
        y = self.heart_train['heart attack risk (binary)']
        X = self.heart_train.drop(['unnamed: 0', 'id', 'heart attack risk (binary)'], axis=1)
        X_heart_test = self.heart_test.drop(['unnamed: 0', 'id'], axis=1)
        ids = self.heart_test[['id']]
        return X, y, X_heart_test, ids




class GBC:
    def __init__(self, learning_rate=0.2, max_depth=2,
                                            n_estimators=200,
                                            random_state=42):
        self.model = SklearnGBC(learning_rate = learning_rate,
                                             max_depth = max_depth,
                                              n_estimators = n_estimators,
                                               random_state = random_state )
    def fit(self, X, y):
        self.model.fit(X, y)  

    def predict(self, X_heart_test, thresholds = 0.30):
        
        result = self.model.predict_proba(X_heart_test)[:, 1]
        result_final = (result >= thresholds).astype('int')
        return result_final    
        
data_loader = LoadData()
data_loader.fill_na_lower()

splitter = split_heart()
X, y, X_heart_test, ids = splitter.split()

model = GBC()
model.fit(X, y)

@app.get("/")
def predict_risk():
    predictions = model.predict(X_heart_test)
    df_res = pd.DataFrame({
        'id': ids['id'],
        'prediction': predictions
    })
    results = df_res.to_dict(orient='records')
    return {'predictions': results}  # без .tolist()

@app.post("/predict")
async def predict_risk_ids():
    try:
        predictions = model.predict(X_heart_test)
        df_res = pd.DataFrame({
            'id': ids['id'],
            'prediction': predictions
        })
        results = df_res.to_dict(orient='records')
        return {'predictions_ids': results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
   