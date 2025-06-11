import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import calplot

# Configuración de la app
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

# 🔬 ENTRENAMIENTO DEL MODELO RANDOM FOREST
df_modelo = df.copy()
df_modelo["condicion_simplificada"] = df_modelo["condicion_simplificada"].astype("category").cat.codes

features = ["tempmax", "humidity", "condicion_simplificada"]
X = df_modelo[features]
y = df_modelo["uvindex"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Sidebar de filtros
st.sidebar.header("📅 Filtros")
año_sel = st.sidebar.selectbox("Selecciona un año", sorted(df["año"].unique()), index=0)
mes_sel = st.sidebar.selectbox("Selecciona un mes", sorted(df["mes"].unique()), index=0)
df_filtrado = df[(df["año"] == año_sel) & (df["mes"] == mes_sel)]

# 🔎 Explicación de riesgos
with st.expander("ℹ️ ¿Qué significa cada tipo de riesgo?"):
    st.markdown("""
    - **Riesgo UV alto**: índice UV ≥ 8. Puede provocar quemaduras en menos de 30 minutos sin protección.
    - **Riesgo por calor extremo**: temperatura máxima ≥ 35°C.
    - **Riesgo por humedad alta**: humedad relativa ≥ 85%.
    """)

# 📈 Evolución UV
st.subheader("📈 Evolución diaria del índice UV")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df_filtrado["datetime"], df_filtrado["uvindex"], marker='o', color='orange')
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Índice UV")
ax1.set_title("Índice UV diario")
ax1.grid(True)
st.pyplot(fig1)

# 🔥 Tipos de riesgo
st.subheader("🔥 Cantidad de días con riesgo climático")
riesgos = df_filtrado[["riesgo_uv", "riesgo_calor", "riesgo_humedad"]].sum()
riesgos.index = ["UV alto", "Calor extremo", "Humedad alta"]
fig2, ax2 = plt.subplots()
sns.barplot(x=riesgos.index, y=riesgos.values, ax=ax2, palette="Reds")
ax2.set_ylabel("Número de días")
st.pyplot(fig2)

# 🌦️ Condiciones más comunes
st.subheader("🌦️ Condiciones meteorológicas más frecuentes")
condiciones = df_filtrado["condicion_simplificada"].value_counts()
fig3, ax3 = plt.subplots()
sns.barplot(x=condiciones.index, y=condiciones.values, ax=ax3, palette="Blues")
ax3.set_ylabel("Número de días")
st.pyplot(fig3)

# 📋 Top 10 días con más riesgo
st.subheader("📋 Top 10 días con mayor riesgo")
tabla = df_filtrado.sort_values(by="riesgo_total", ascending=False).head(10)
st.dataframe(tabla[["datetime", "uvindex", "tempmax", "humidity", "riesgo_total", "condicion_simplificada"]])

# 🗓️ Calendario de riesgo total
st.subheader("🗓️ Calendario de riesgo total por día")
riesgo_diario = df.groupby(df["datetime"].dt.date)["riesgo_total"].sum()
riesgo_diario = pd.Series(riesgo_diario)
riesgo_diario.index = pd.to_datetime(riesgo_diario.index)
fig_cal = calplot.calplot(riesgo_diario, cmap="Reds", colorbar=True, suptitle="Nivel de riesgo diario (0 a 3)")
st.pyplot(fig_cal)

# 🎛️ Predicción personalizada
st.subheader("🎛️ Predicción personalizada del índice UV")
st.markdown("""
¿Te has preguntado cuánto podría aumentar el riesgo solar en función del clima?  
Con esta herramienta puedes **simular el índice UV** esperado introduciendo condiciones meteorológicas reales o hipotéticas.

Esto puede ayudarte a:
- Evaluar el riesgo de quemaduras o exposición solar.
- Tomar decisiones informadas sobre protección solar.
- Planificar actividades al aire libre con mayor seguridad.

### 🧠 ¿Por qué es posible predecir el índice UV con estos datos?
- **Temperatura máxima**: Se asocia a mayor radiación solar.
- **Humedad**: Afecta la dispersión de radiación.
- **Condición meteorológica**: Nubes, lluvia, etc. modifican la exposición solar.
""")

col1, col2 = st.columns(2)
with col1:
    temp_input = st.number_input("Temperatura máxima (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity_input = st.slider("Humedad relativa (%)", 0, 100, 60)
with col2:
    condicion_input = st.selectbox("Condición meteorológica", ["despejado", "nublado", "lluvia", "otros"])
cond_map = {"despejado": 0, "lluvia": 1, "nublado": 2, "otros": 3}
cond_codificada = cond_map.get(condicion_input, 3)
if st.button("Predecir índice UV"):
    entrada = pd.DataFrame([[temp_input, humidity_input, cond_codificada]], columns=features)
    prediccion = modelo.predict(entrada)[0]
    st.success(f"🌞 El índice UV estimado es: **{prediccion:.2f}**")

# 📊 Evaluación del modelo
st.subheader("🔬 Evaluación del modelo Random Forest")
st.markdown(f"**RMSE:** {sqrt(mean_squared_error(y_test, y_pred)):.2f} &nbsp;&nbsp;&nbsp; **R²:** {r2_score(y_test, y_pred):.2f}")

fig_rf1, ax_rf1 = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax_rf1)
ax_rf1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax_rf1.set_xlabel("Valor real")
ax_rf1.set_ylabel("Valor predicho")
ax_rf1.set_title("Random Forest – Índice UV")
st.pyplot(fig_rf1)

importancia = pd.Series(modelo.feature_importances_, index=features)
fig_rf2, ax_rf2 = plt.subplots()
sns.barplot(x=importancia.index, y=importancia.values, ax=ax_rf2)
ax_rf2.set_title("Importancia de variables")
ax_rf2.set_ylabel("Peso")
st.pyplot(fig_rf2)
