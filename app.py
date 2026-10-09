import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import KNNImputer

# Configuración de la página
st.set_page_config(page_title="Predicción de Calidad del Aire (PM2.5)", layout="centered")
st.title("💨 Predicción de Calidad del Aire (PM2.5)")
st.write("Introduce los datos meteorológicos y de ubicación para predecir la concentración de PM2.5 usando el modelo SVM optimizado con PCA.")

# 1. Cargar transformadores y modelo preentrenados
@st.cache_resource
def load_assets():
    encoder = joblib.load('one_hot_encoder.joblib')
    scaler = joblib.load('minmax_scaler.joblib')
    pca = joblib.load('pca_transformer.joblib')
    model = joblib.load('svm_optimized_model.joblib')
    return encoder, scaler, pca, model

try:
    encoder, scaler, pca, model = load_assets()
    st.success("✅ Modelos y transformadores cargados con éxito.")
except Exception as e:
    st.error(f"❌ Error al cargar los archivos .joblib: {e}")

# Formulario de entrada de datos
with st.form("prediction_form"):
    st.subheader("📊 Datos de Entrada")
    
    col1, col2 = st.columns(2)
    with col1:
        temp_c = st.number_input("Temperatura (°C)", min_value=-10.0, max_value=50.0, value=20.0)
        humidity = st.number_input("Humedad (%)", min_value=0.0, max_value=100.0, value=60.0)
        wind_speed = st.number_input("Velocidad del Viento (m/s)", min_value=0.0, max_value=30.0, value=5.0)
        pressure = st.number_input("Presión Atmosférica (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)
    
    with col2:
        rain = st.number_input("Precipitación/Lluvia (mm)", min_value=0.0, max_value=100.0, value=0.0)
        hour = st.slider("Hora del Día", min_value=0, max_value=23, value=12)
        day_of_week = st.slider("Día de la Semana (0=Lunes, 6=Domingo)", min_value=0, max_value=6, value=2)
        month = st.slider("Mes del Año", min_value=1, max_value=12, value=6)

    st.subheader("📍 Ubicación y Dirección del Viento")
    col3, col4 = st.columns(2)
    with col3:
        # Opciones disponibles en base al codificador
        estaciones = [f"Station_{i}" for i in range(1, 25)]
        station = st.selectbox("Estación de Monitoreo", estaciones)
    with col4:
        direcciones = ['E', 'N', 'NE', 'NW', 'S', 'SE', 'SW', 'W']
        wind_direction = st.selectbox("Dirección del Viento", direcciones)

    submit_button = st.form_submit_button(label="🔮 Calcular PM2.5")

if submit_button:
    try:
        # 1. Crear DataFrame con la estructura original
        input_data = pd.DataFrame({
            'Temperature_C': [temp_c],
            'Humidity': [humidity],
            'WindSpeed': [wind_speed],
            'Pressure': [pressure],
            'Rain': [rain],
            'Hour': [float(hour)],
            'DayOfWeek': [float(day_of_week)],
            'Month': [float(month)],
            'Station': [station],
            'WindDirection': [wind_direction]
        })
        
        # 2. Aplicar One Hot Encoding a las variables categóricas
        encoded_cats = encoder.transform(input_data[['Station', 'WindDirection']])
        encoded_cols = encoder.get_feature_names_out(['Station', 'WindDirection'])
        df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=input_data.index)
        
        # Combinar y eliminar originales
        input_processed = pd.concat([input_data.drop(columns=['Station', 'WindDirection']), df_encoded], axis=1)
        
        # 3. Escalar usando el MinMaxScaler preentrenado
        features_expected = scaler.feature_names_in_
        input_processed = input_processed[features_expected] # Alinear columnas
        input_scaled = scaler.transform(input_processed)
        
        # 4. Transformar con PCA
        input_pca = pca.transform(input_scaled)
        
        # 5. Predicción con SVM
        pred_value = model.predict(input_pca)[0]
        
        # Mostrar Resultado
        st.subheader("🎉 Resultado de la Predicción")
        st.metric(label="Concentración Estimada de PM2.5", value=f"{pred_value:.2f} µg/m³")
        
        # Indicador visual simple de calidad de aire
        if pred_value <= 12:
            st.success("🟢 Calidad del Aire: Buena")
        elif pred_value <= 35.4:
            st.warning("🟡 Calidad del Aire: Moderada")
        else:
            st.error("🔴 Calidad del Aire: Dañina / No Saludable")
            
    except Exception as e:
        st.error(f"Ha ocurrido un error en el preprocesamiento o predicción: {e}")
