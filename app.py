# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calplot

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



