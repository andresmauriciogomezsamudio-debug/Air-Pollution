import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# Configuración de la página
st.set_page_config(page_title="Predicción de Calidad del Aire (PM2.5)", layout="centered")
st.title("💨 Predicción de Calidad del Aire (PM2.5)")
st.write("Ingresa los datos meteorológicos manualmente o sube un archivo con múltiples registros para obtener predicciones inmediatas.")

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

# Crear pestañas para separar la predicción individual de la predicción por lote
tab1, tab2 = st.tabs(["Individual", "Carga de Archivo (Excel/CSV)"])

with tab1:
    # Formulario de entrada de datos individuales
    with st.form("prediction_form"):
        st.subheader("📊 Datos de Entrada Manual")
        
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
            estaciones = [f"Station_{i}" for i in range(1, 25)]
            station = st.selectbox("Estación de Monitoreo", estaciones)
        with col4:
            direcciones = ['E', 'N', 'NE', 'NW', 'S', 'SE', 'SW', 'W']
            wind_direction = st.selectbox("Dirección del Viento", direcciones)

        submit_button = st.form_submit_button(label="🔮 Calcular PM2.5")

    if submit_button:
        try:
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
            
            encoded_cats = encoder.transform(input_data[['Station', 'WindDirection']])
            encoded_cols = encoder.get_feature_names_out(['Station', 'WindDirection'])
            df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=input_data.index)
            
            input_processed = pd.concat([input_data.drop(columns=['Station', 'WindDirection']), df_encoded], axis=1)
            features_expected = scaler.feature_names_in_
            input_processed = input_processed[features_expected]
            input_scaled = scaler.transform(input_processed)
            input_pca = pca.transform(input_scaled)
            pred_value = model.predict(input_pca)[0]
            
            st.subheader("🎉 Resultado")
            st.metric(label="Concentración Estimada de PM2.5", value=f"{pred_value:.2f} µg/m³")
            
            if pred_value <= 12:
                st.success("🟢 Calidad del Aire: Buena")
            elif pred_value <= 35.4:
                st.warning("🟡 Calidad del Aire: Moderada")
            else:
                st.error("🔴 Calidad del Aire: Dañina / No Saludable")
                
        except Exception as e:
            st.error(f"Error en la predicción manual: {e}")

with tab2:
    st.subheader("📁 Carga un Archivo de Datos")
    st.write("Sube un archivo de Excel (`.xlsx`, `.xls`) o CSV con la estructura de variables requerida.")
    
    uploaded_file = st.file_uploader("Seleccionar archivo", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Determinar tipo de archivo y cargarlo en un DataFrame
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
                
            st.write("**Vista previa de los datos subidos:**")
            st.dataframe(df_upload.head(10))
            
            # Requisitos mínimos de columnas para proceder
            columnas_requeridas = ['Temperature_C', 'Humidity', 'WindSpeed', 'Pressure', 'Rain', 'Hour', 'DayOfWeek', 'Month', 'Station', 'WindDirection']
            
            missing_cols = [col for col in columnas_requeridas if col not in df_upload.columns]
            if len(missing_cols) > 0:
                st.error(f"❌ Al archivo cargado le faltan las siguientes columnas: {missing_cols}")
            else:
                if st.button("⚡ Generar Predicciones Masivas"):
                    with st.spinner("Procesando lote y calculando predicciones..."):
                        # Trabajar con las columnas requeridas limpias
                        df_eval = df_upload[columnas_requeridas].copy()
                        
                        # Aplicar One Hot Encoding
                        encoded_cats = encoder.transform(df_eval[['Station', 'WindDirection']])
                        encoded_cols = encoder.get_feature_names_out(['Station', 'WindDirection'])
                        df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=df_eval.index)
                        
                        # Combinar y ordenar columnas según el scaler
                        input_processed = pd.concat([df_eval.drop(columns=['Station', 'WindDirection']), df_encoded], axis=1)
                        features_expected = scaler.feature_names_in_
                        input_processed = input_processed[features_expected]
                        
                        # Escalar, aplicar PCA y predecir
                        input_scaled = scaler.transform(input_processed)
                        input_pca = pca.transform(input_scaled)
                        lote_predictions = model.predict(input_pca)
                        
                        # Guardar predicciones en la copia final
                        df_results = df_upload.copy()
                        df_results['Predicted_PM25'] = lote_predictions
                        
                        st.success("🎉 Predicciones procesadas con éxito.")
                        st.write("**Vista previa de los resultados:**")
                        st.dataframe(df_results.head(10))
                        
                        # Botón para descargar archivo procesado
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_results.to_excel(writer, index=False, sheet_name='Predicciones')
                        processed_data = output.getvalue()
                        
                        st.download_button(
                            label="📥 Descargar Excel con Predicciones",
                            data=processed_data,
                            file_name="predicciones_pm25_lote.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")
