# ----------- Importaciones

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ----------- Fin de las Importaciones

# 1. Abre y examina el archivo de datos. Dirección al archivo
df = pd.read_csv(r"C:\\Users\\sasor\\Desktop\\Tripleten\\Sprint 10\\Proyecto\\users_behavior.csv")

"""esto nos permite revisar que se haya cargado correctamente"""
print(df.head())

"""aqui se verifica el tipo de datos de cada una"""
print(df.info())
print(df.dtypes)

# 2. Segmenta los datos fuente en un conjunto de entrenamiento, uno de validación y uno de prueba.

"""dado que se reconoce que todas las columnas tienen el tipo de dato correcto se procede directo sin hacer alguna limpieza del df"""

# dividir todo en 3 partes con proporción 3:1:1 - (60% entrenamiento, 20% validación, 20% prueba).

features = df.drop(["is_ultra"], axis=1)
target = df["is_ultra"]

features_train, features_rest, target_train, target_rest = train_test_split(features, target, test_size=0.4, random_state=12345)

features_valid, features_test, target_valid, target_test = train_test_split(
    features_rest, target_rest, test_size=0.5, random_state=12345)

"""Verificacion de que las proporciones de clase se mantuvieron"""

print("Entrenamiento:", features_train.shape, target_train.shape)
print("Validación", features_valid.shape, target_valid.shape)
print("Prueba", features_test.shape, target_test.shape)

print("Entrenamiento", target_train.mean())
print("Validación", target_valid.mean())
print("Prueba", target_test.mean())

# 2. Investiga la calidad de diferentes modelos cambiando los hiperparámetros. Describe brevemente los hallazgos del estudio.

