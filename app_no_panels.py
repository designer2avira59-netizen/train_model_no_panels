# app_no_panels.py
import streamlit as st
import xgboost as xgb
import os

# --- Загрузка модели ---
@st.cache_resource
def load_model():
    model_path = 'frame_model_no_panels.json'
    if not os.path.exists(model_path):
        st.error("❌ Модель не найдена. Загрузите frame_model_no_panels.json в эту папку.")
        st.stop()
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

model = load_model()

# --- Интерфейс ---
st.title("🧱 Оценка каркаса (без учёта щитов)")
st.write("Введите параметры — получите прогноз. Не нужно знать количество щитов.")

base_area = st.number_input("Площадь основания (м²)", 10.0, 300.0, 123.0)
height = st.number_input("Высота (м)", 2.0, 10.0, 5.95)
num_floors = st.number_input("Этажей", 1, 6, 4)
total_post_length = st.number_input("Суммарная длина стоек (м)", 10.0, 1000.0, 284.0)
total_beam_length = st.number_input("Суммарная длина перемычек (м)", 10.0, 1000.0, 440.0)
shape_type = st.selectbox(
    "Форма каркаса",
    options=[0, 1, 2, 4, 5],
    format_func=lambda x: {
        0: "0 - Куб",
        1: "1 - O-образный",
        2: "2 - П-образный",
        4: "4 - Г-образный",
        5: "5 - Линия"
    }[x]
)

# --- Кнопка расчёта ---
if st.button("🧮 Рассчитать стоимость"):
    from datetime import datetime
    start_date = datetime(2025, 7, 1)
    today = datetime.today()
    days = (today - start_date).days

    input_data = [[base_area, height, num_floors, total_post_length, total_beam_length, shape_type, days]]

    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"🎯 **Прогноз: {predicted_price:,.0f} руб.**")
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")