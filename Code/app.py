import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

df = pd.read_csv(r'C:\Users\user\Desktop\Lahiye\dataset\predictive_maintenance.csv')
df = df.drop(['UDI', 'Product ID'], axis=1)
X = df.drop(['Target', 'Failure Type'], axis=1)
y = df['Target']

categorical_features = ['Type']
numerical_features = X.columns.difference(categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f'Model Accuracy: {accuracy_score(y_test, y_pred)}')


with open('predictive_maintenance_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

app = FastAPI()

class PredictRequest(BaseModel):
    Type: str
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: int
    Torque_Nm: float
    Tool_wear_min: int


@app.post("/predict")
async def predict(request: PredictRequest):
    request_dict = request.dict()
    input_df = pd.DataFrame([request_dict])
    prediction = model.predict(input_df)[0]
    return {"prediction": prediction}

import nest_asyncio
nest_asyncio.apply()

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5001, log_level="debug")


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)