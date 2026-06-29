# ----------- Importaciones

import pandas as pd

# ----------- Fin de las Importaciones

# 1. Abre y examina el archivo de datos. Dirección al archivo
df = pd.read_csv(r"C:\\Users\\sasor\\Desktop\\Tripleten\\Sprint 10\\Proyecto\\users_behavior.csv")

"""esto nos permite revisar que se haya cargado correctamente"""
print(df.head())

"""aqui se verifica el tipo de datos de cada una"""
print(df.info())
print(df.dtypes)
