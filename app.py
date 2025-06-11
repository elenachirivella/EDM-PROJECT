# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


# Configuración
st.set_page_config(page_title="Riesgo Climático en Valencia", layout="wide")

# Título
st.title("🌤️ Riesgo Climático Diario en Valencia")
st.markdown("""
Esta aplicación muestra la evolución de los riesgos climáticos en Valencia basados en tres factores: **índice UV elevado**, **calor extremo** y **humedad alta**.
""")

# Cargar datos
df = pd.read_csv("datos_valencia_limpios.csv", parse_dates=["datetime"])

# Añadir columnas de mes y año
df["mes"] = df["datetime"].dt.month
df["año"] = df["datetime"].dt.year

# Sidebar de filtros
st.sidebar.header("📅 Filtros")
año_sel = st.sidebar.selectbox("Selecciona un año", sorted(df["año"].unique()), index=0)
mes_sel = st.sidebar.selectbox("Selecciona un mes", sorted(df["mes"].unique()), index=0)

df_filtrado = df[(df["año"] == año_sel) & (df["mes"] == mes_sel)]

# 🔎 Explicación de riesgos
with st.expander("ℹ️ ¿Qué significa cada tipo de riesgo?"):
    st.markdown("""
    - **Riesgo UV alto**: índice UV ≥ 8. Puede provocar quemaduras en menos de 30 minutos sin protección.
    - **Riesgo por calor extremo**: temperatura máxima ≥ 35°C. Aumenta el riesgo de golpe de calor.
    - **Riesgo por humedad alta**: humedad relativa ≥ 85%. Dificulta la evaporación del sudor y aumenta la sensación térmica.
    """)

# Gráfico: evolución del índice UV
st.subheader("📈 Evolución diaria del índice UV")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df_filtrado["datetime"], df_filtrado["uvindex"], marker='o', color='orange')
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Índice UV")
ax1.set_title("Índice UV diario")
ax1.grid(True)
st.pyplot(fig1)

# Gráfico: días con cada tipo de riesgo
st.subheader("🔥 Cantidad de días con riesgo climático")
riesgos = df_filtrado[["riesgo_uv", "riesgo_calor", "riesgo_humedad"]].sum()
riesgos.index = ["UV alto", "Calor extremo", "Humedad alta"]
fig2, ax2 = plt.subplots()
sns.barplot(x=riesgos.index, y=riesgos.values, ax=ax2, palette="Reds")
ax2.set_ylabel("Número de días")
ax2.set_title("Tipos de riesgo climático")
st.pyplot(fig2)

# Gráfico: condiciones meteorológicas más frecuentes
st.subheader("🌦️ Condiciones meteorológicas más frecuentes")
condiciones = df_filtrado["condicion_simplificada"].value_counts()
fig3, ax3 = plt.subplots()
sns.barplot(x=condiciones.index, y=condiciones.values, ax=ax3, palette="Blues")
ax3.set_ylabel("Número de días")
ax3.set_title("Condiciones más comunes")
st.pyplot(fig3)

# Tabla: días con mayor riesgo total
st.subheader("📋 Top 10 días con mayor riesgo")
tabla = df_filtrado.sort_values(by="riesgo_total", ascending=False).head(10)
st.dataframe(tabla[["datetime", "uvindex", "tempmax", "humidity", "riesgo_total", "condicion_simplificada"]])


# 🎛️ PREDICCIÓN EN TIEMPO REAL
st.subheader("🎛️ Predicción personalizada del índice UV")
st.markdown("""
¿Te has preguntado cuánto podría aumentar el riesgo solar en función del clima?  
Con esta herramienta puedes **simular el índice UV** esperado introduciendo condiciones meteorológicas reales o hipotéticas.

Esto puede ayudarte a:
- Evaluar el riesgo de quemaduras o exposición solar.
- Tomar decisiones informadas sobre protección solar.
- Planificar actividades al aire libre con mayor seguridad.

---

### 🧠 ¿Por qué es posible predecir el índice UV con estos datos?

Aunque el índice UV depende principalmente de factores astronómicos y atmosféricos (como la posición del sol y la capa de ozono), **las condiciones locales influyen directamente en su efecto sobre la salud**. Estas variables ayudan a estimar mejor el riesgo:

- **Temperatura máxima**: A menudo asociada a cielos despejados y mayor radiación solar.
- **Humedad**: Afecta la dispersión de la radiación y puede influir en la sensación térmica.
- **Condición meteorológica**: Días nublados o lluviosos suelen reducir la exposición directa, pero no la eliminan por completo.

El modelo ha sido entrenado con datos reales de Valencia para aprender la relación entre estas condiciones y el índice UV registrado.
""")


st.markdown("Introduce condiciones meteorológicas para estimar el índice UV:")

# Formulario de entrada
col1, col2 = st.columns(2)

with col1:
    temp_input = st.number_input("Temperatura máxima (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity_input = st.slider("Humedad relativa (%)", 0, 100, 60)

with col2:
    condicion_input = st.selectbox(
        "Condición meteorológica",
        ["despejado", "nublado", "lluvia", "otros"]
    )

# Codificar la condición como en el modelo
cond_map = {"despejado": 0, "lluvia": 1, "nublado": 2, "otros": 3}
cond_codificada = cond_map.get(condicion_input, 3)

# Predecir
if st.button("Predecir índice UV"):
    entrada = pd.DataFrame([[temp_input, humidity_input, cond_codificada]],
                           columns=["tempmax", "humidity", "condicion_simplificada"])
    prediccion = modelo.predict(entrada)[0]
    st.success(f"🌞 El índice UV estimado es: **{prediccion:.2f}**")

# 🔬 MODELO DE PREDICCIÓN – RANDOM FOREST
st.subheader("🔬 Predicción de índice UV – Modelo Random Forest")
st.markdown("""
Para predecir el índice UV a partir de variables meteorológicas, se ha optado por entrenar un modelo de tipo Random Forest por las siguientes razones:

Es un modelo robusto y versátil, ideal cuando se trabaja con datasets pequeños o medianos como este.

- Permite capturar relaciones no lineales entre las variables. Por ejemplo, el efecto de la humedad sobre el índice UV no siempre es proporcional ni directo.
- Tolera bien los datos con ruido o pequeñas imprecisiones, sin requerir una limpieza excesiva o supuestos estadísticos estrictos.
- Calcula la importancia de cada variable automáticamente, lo que permite entender qué factores influyen más en la predicción (por ejemplo, en este caso, la temperatura tiene mayor peso que la humedad o la condición del cielo).
- Es fácil de interpretar visualmente, lo que resulta ideal para una aplicación educativa y divulgativa como esta.

En resumen, Random Forest es una opción equilibrada entre precisión, facilidad de entrenamiento y explicabilidad, lo que lo convierte en una elección adecuada para este tipo de aplicación ciudadana basada en datos abiertos.

""")

# Preparar datos para el modelo
df_modelo = df.copy()
df_modelo["condicion_simplificada"] = df_modelo["condicion_simplificada"].astype("category").cat.codes

features = ["tempmax", "humidity", "condicion_simplificada"]
X = df_modelo[features]
y = df_modelo["uvindex"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Evaluación
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Gráfico real vs predicho
fig_rf1, ax_rf1 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax_rf1)
ax_rf1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax_rf1.set_xlabel("Valor real")
ax_rf1.set_ylabel("Valor predicho")
ax_rf1.set_title("Random Forest – Índice UV")
st.pyplot(fig_rf1)

# Importancia de variables
importancia = pd.Series(modelo.feature_importances_, index=features)
fig_rf2, ax_rf2 = plt.subplots()
sns.barplot(x=importancia.index, y=importancia.values, ax=ax_rf2)
ax_rf2.set_title("Importancia de variables")
ax_rf2.set_ylabel("Peso")
st.pyplot(fig_rf2)
