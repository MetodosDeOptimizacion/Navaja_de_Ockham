import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Crear el dataset ficticio
data = {
    'Tamaño': [100, 150, 80, 120, 200, 90],
    'Habitaciones': [3, 4, 2, 3, 5, 2],
    'Baños': [2, 3, 1, 2, 4, 1],
    'Antigüedad': [10, 5, 20, 8, 2, 15],
    'Precio': [200000, 300000, 150000, 250000, 450000, 180000]
}

df = pd.DataFrame(data)

# Separar características y target
X_simple = df[['Tamaño']]  # Solo tamaño
X_complex = df[['Tamaño', 'Habitaciones', 'Baños', 'Antigüedad']]  # Todas las características
y = df['Precio']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_complex, y, test_size=0.2, random_state=42)

# Entrenar el Modelo A (simple)
model_simple = LinearRegression()
model_simple.fit(X_train[['Tamaño']], y_train)

# Entrenar el Modelo B (complejo)
model_complex = LinearRegression()
model_complex.fit(X_train, y_train)

# Evaluar ambos modelos
y_pred_simple = model_simple.predict(X_test[['Tamaño']])
y_pred_complex = model_complex.predict(X_test)

mse_simple = mean_squared_error(y_test, y_pred_simple)
mse_complex = mean_squared_error(y_test, y_pred_complex)

print(f"MSE del Modelo Simple (A): {mse_simple}")
print(f"MSE del Modelo Complejo (B): {mse_complex}")
