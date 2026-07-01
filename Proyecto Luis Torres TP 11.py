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

"""arbol de decision"""

best_tree_score = 0
best_tree_depth = 0
for depth in range(1, 11):
    model = DecisionTreeClassifier(random_state=12345, max_depth=depth)
    model.fit(features_train, target_train)
    score = model.score(features_valid, target_valid)
    print(f"max_depth = {depth}: exactitud en validacion = {score:.4f}")
    if score > best_tree_score:
        best_tree_score = score
        best_tree_depth = depth

print(f"Mejor arbol de decision es max_depth={best_tree_depth},"
      f"exactitud en validacion={best_tree_score:.4f}")
 
"""bosque aleatorio"""

best_forest_score = 0
best_forest_est = 0
best_forest_depth = 0
for est in range(10, 101, 10):
    for depth in range(1, 11):
        model = RandomForestClassifier(
            random_state=12345, n_estimators=est, max_depth=depth)
        model.fit(features_train, target_train)
        score = model.score(features_valid, target_valid)
        if score > best_forest_score:
            best_forest_score = score
            best_forest_est = est
            best_forest_depth = depth
 
print(f"Mejor bosque aleatorio -> n_estimators={best_forest_est},"
      f"max_depth={best_forest_depth},"
      f"exactitud en validacion={best_forest_score:.4f}")
 
"""regresion logistica"""

log_model = LogisticRegression(random_state=12345, solver="liblinear")
log_model.fit(features_train, target_train)
log_score_train = log_model.score(features_train, target_train)
log_score_valid = log_model.score(features_valid, target_valid)
print(f"Exactitud en entrenamiento: {log_score_train:.4f}")
print(f"Exactitud en validación: {log_score_valid:.4f}")
 
# 4. Comprueba la calidad del modelo usando el conjunto de prueba.

final_model = RandomForestClassifier(
    random_state=12345, n_estimators=best_forest_est, max_depth=best_forest_depth)
final_model.fit(features_train, target_train)
 
test_predictions = final_model.predict(features_test)
test_accuracy = accuracy_score(target_test, test_predictions)
 
"""evaluacion del modelo"""
print(f'modelo final: RandomForestClassifier(n_estimators={best_forest_est},'
      f'max_depth={best_forest_depth})')
print(f'exactitud:{test_accuracy:.4f}')

if test_accuracy >= 0.75:
    print("el modelo tiene un umbral igual o mayor de exactitud a 0.75")
else:
    print("El modelo tiene un umbral menor de exactitud")
    
