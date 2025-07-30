import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st
import pickle


# 1. Загрузка и предобработка данных
@st.cache_data
def load_data():
    df = pd.read_csv('realty_data.csv')

    # Преобразование числовых полей
    df['price'] = df['price'].astype(float)
    df['total_square'] = df['total_square'].astype(float)

    # Заполнение пропусков
    df['rooms'] = df['rooms'].fillna(0).astype(int)
    df['floor'] = df['floor'].fillna(1).astype(int)

    return df


df = load_data()

# 2. Подготовка признаков и целевой переменной
features = ['total_square', 'rooms', 'floor', 'object_type', 'city']
X = df[features]
y = df['price']


# 3. Создание и обучение модели
@st.cache_resource
def train_model():
    # Препроцессинг для категориальных признаков
    categorical_features = ['object_type', 'city']
    numerical_features = ['total_square', 'rooms', 'floor']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X, y)
    return model


model = train_model()

# 4. Сохранение модели (для демонстрации)
with open('realty_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 5. Streamlit интерфейс
st.title('Прогнозирование стоимости недвижимости 🏠')
st.write('Введите параметры недвижимости для оценки стоимости:')

# Форма ввода данных
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        total_square = st.number_input('Общая площадь (м²)', min_value=10.0, max_value=500.0, value=50.0)
        rooms = st.selectbox('Количество комнат', [1, 2, 3, 4, 5])
        floor = st.number_input('Этаж', min_value=1, max_value=50, value=5)

    with col2:
        object_type = st.selectbox('Тип объекта', df['object_type'].unique())
        city = st.selectbox('Город', df['city'].unique())

    submitted = st.form_submit_button("Оценить стоимость")

# 6. Обработка и вывод результата
if submitted:
    input_data = pd.DataFrame([[total_square, rooms, floor, object_type, city]],
                              columns=features)

    prediction = model.predict(input_data)[0]

    st.success(f'### Прогнозируемая стоимость: {prediction:,.0f} руб.')
    st.balloons()

    # Дополнительная визуализация
    st.subheader('Сравнение с похожими объектами')
    similar = df[
        (df['object_type'] == object_type) &
        (df['city'] == city) &
        (df['rooms'] == rooms)
        ].sort_values('total_square')

    if not similar.empty:
        st.dataframe(similar[['total_square', 'price', 'address_name']])
    else:
        st.warning('В базе нет похожих объектов для сравнения')

# 7. Отображение сырых данных
if st.checkbox('Показать исходные данные'):
    st.write(df)









