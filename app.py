import pandas as pd

# Cargar el dataset original
df = pd.read_csv("datos_valencia_limpios.csv", sep=None, engine='python')

# Convertir la columna de fecha a formato datetime
df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True)

# Crear variables de riesgo
df["riesgo_uv"] = df["uvindex"] >= 8
df["riesgo_calor"] = df["tempmax"] >= 35
df["riesgo_humedad"] = df["humidity"] >= 85

# Crear una columna total de riesgo (suma de condiciones verdaderas)
df["riesgo_total"] = df[["riesgo_uv", "riesgo_calor", "riesgo_humedad"]].sum(axis=1)

# Clasificar condiciones meteorológicas
df["condicion_simplificada"] = df["conditions"].str.lower().map(
    lambda x: "despejado" if "clear" in x else
              "nublado" if "cloud" in x else
              "lluvia" if "rain" in x else
              "otros"
)

# Guardar un resumen de las nuevas columnas
columnas_nuevas = df[["datetime", "uvindex", "tempmax", "humidity",
                      "riesgo_uv", "riesgo_calor", "riesgo_humedad",
                      "riesgo_total", "condicion_simplificada"]]

# Guardar en un nuevo CSV
output_path = "/mnt/data/datos_valencia_limpios.csv"
columnas_nuevas.to_csv(output_path, index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="Datos Valencia Limpios", dataframe=columnas_nuevas)

output_path
