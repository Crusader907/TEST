
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

# Инициализация FastAPI
app = FastAPI(title="Real Estate Price Predictor")


# Загрузка и подготовка данных
def load_data():
    df = pd.read_csv("realty_data.csv")
    df['price'] = df['price'].astype(float)
    df['total_square'] = df['total_square'].astype(float)
    df['rooms'] = df['rooms'].fillna(0).astype(int)
    df['floor'] = df['floor'].fillna(1).astype(int)
    return df


# Обучение модели
def train_model():
    df = load_data()
    features = ['total_square', 'rooms', 'floor', 'object_type', 'city']
    X = df[features]
    y = df['price']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['total_square', 'rooms', 'floor']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['object_type', 'city'])
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X, y)

    # Сохраняем модель
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model


# Загружаем или обучаем модель
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    model = train_model()


# Модель для POST-запроса
class PredictionRequest(BaseModel):
    total_square: float
    rooms: int
    floor: int
    object_type: str
    city: str


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "OK", "message": "Service is running"}


# GET endpoint
@app.get("/predict_get")
def predict_get(total_square: float, rooms: int, floor: int, object_type: str, city: str):
    input_data = pd.DataFrame([[total_square, rooms, floor, object_type, city]],
                              columns=['total_square', 'rooms', 'floor', 'object_type', 'city'])
    prediction = model.predict(input_data)[0]
    return {"predicted_price": round(prediction, 2)}


# POST endpoint
@app.post("/predict_post")
def predict_post(request: PredictionRequest):
    input_data = pd.DataFrame([[request.total_square, request.rooms, request.floor,
                                request.object_type, request.city]],
                              columns=['total_square', 'rooms', 'floor', 'object_type', 'city'])
    prediction = model.predict(input_data)[0]
    return {"predicted_price": round(prediction, 2)}

# Запуск сервера (выполнить в терминале)
# uvicorn task_10:app --reload




