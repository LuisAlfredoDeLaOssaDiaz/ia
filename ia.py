# %%
#!pip install pandas

# %%
import os
import io
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
import missingno as msno
from imblearn.over_sampling import RandomOverSampler
import plotly.express as px
from collections import Counter
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV,RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm



# %%
file_path = "/home/luis/Documentos/Inteligencia_artificial/data.csv"

try:
    with open(file_path, 'r') as file:
        # Realizar operaciones con el archivo aquí
        # Por ejemplo, puedes leer su contenido o procesarlo de alguna manera
        uploaded = file.read()
        print("Contenido del archivo:")
        print(uploaded)
except FileNotFoundError:
    print("El archivo no fue encontrado en la ruta especificada.")
except Exception as e:
    print("Se produjo un error al abrir el archivo:", str(e))


# %%
filename = '/home/luis/Documentos/Inteligencia_artificial/data.csv'
df = pd.read_csv(filename, delimiter=";")
print(df)
pd.set_option('display.max_columns', None)
df.head()


# %%
df.describe()

# %%
df.info()

# %%
sns.clustermap(df.corr(), cmap = "vlag", dendrogram_ratio = (0.1, 0.1), annot = True, linewidths = .8, figsize = (9,10))
plt.show()

# %%
df.isnull().sum()

# %%
X = df.drop('Target (Total orders)', axis=1)
y = df['Target (Total orders)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)
print("Forma de X_train:", X_train.shape)
print("Forma de X_test:", X_test.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de y_test:", y_test.shape)

# %%
x_train_max = np.max(X_train, axis=0)
x_train_min = np.min(X_train, axis=0)
X_train_normalized = (X_train - x_train_min) / (x_train_max - x_train_min)
X_test_normalized = (X_test - x_train_min) / (x_train_max - x_train_min)
print()
print(x_train_max)
print()
print(x_train_min)
print()
print(X_train_normalized)
print()
print(X_test_normalized)

# %%
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


models = [("GradientBoostingRegressor", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)),
          ("LinearRegression", LinearRegression()),
          ("DecisionTreeRegressor", DecisionTreeRegressor(max_depth=4)),
          ("RandomForestRegressor", RandomForestRegressor())]

finalResults = []
k_fold = ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
for name, model in models:
    mse_scores = []
    r2_scores = []
    mae_scores = []
    for train_index, test_index in k_fold.split(X_train, y_train):
        data_train, data_test = X_train.iloc[train_index], X_train.iloc[test_index]
        target_train, target_test = y_train.iloc[train_index], y_train.iloc[test_index]
        model.fit(data_train, target_train)
        predict = model.predict(data_test)
        mse = mean_squared_error(target_test, predict)
        mse_scores.append(mse)
        r2 = r2_score(target_test, predict)
        r2_scores.append(r2)
        mae = mean_absolute_error(target_test, predict)
        mae_scores.append(mae)
    avg_mse = sum(mse_scores) / len(mse_scores)
    avg_r2 = sum(r2_scores) / len(r2_scores)
    avg_mae = sum(mae_scores) / len(mae_scores)
    finalResults.append({'name': name, 'MSE': avg_mse, 'R^2': avg_r2, 'MAE': avg_mae})
print(finalResults)


# %%
df_result = pd.DataFrame.from_dict(finalResults)
df_result

# %%
model_names = [result['name'] for result in finalResults]
mse_values = [result['MSE'] for result in finalResults]
plt.figure(figsize=(10, 6))
plt.bar(model_names, mse_values, label='MSE', alpha=0.5)
plt.xlabel('Modelos')
plt.ylabel('Valor')
plt.title('Evaluación de Modelos de Regresión')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# %%
model_names = [result['name'] for result in finalResults]
mae_values = [result['MAE'] for result in finalResults]

plt.bar(model_names, mae_values, label='MAE', alpha=0.7,color = "green")

plt.xlabel('Modelos')
plt.ylabel('Valor')
plt.title('Evaluation')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
model_names = [result['name'] for result in finalResults]
r2_values = [result['R^2'] for result in finalResults]
plt.bar(model_names, r2_values, label='R^2', alpha=0.9, color = "red")
plt.xlabel('Modelos')
plt.ylabel('Valor')
plt.title('Evaluation')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Entrenar el modelo de regresión lineal
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

# Hacer predicciones en los datos de prueba
predictions = linear_regression_model.predict(X_test)

# Graficar las predicciones frente a los valores reales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue', label='Predicciones')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Línea de Predicción Ideal')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('LinearRegression')
plt.legend()
plt.show()


