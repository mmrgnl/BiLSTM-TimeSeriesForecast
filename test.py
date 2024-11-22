import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('bilstm_model.h5')

X_test1 = pd.read_csv("X_test1.csv").values
Y_test1 = pd.read_csv("Y_test1.csv").values

X_test = X_test1.reshape(X_test1.shape[0], 1, X_test1.shape[1])

# Предсказания на тестовых данных
y_pred = model.predict(X_test)

# Оценка модели
test_loss = model.evaluate(X_test, Y_test1, verbose=1)
print(f'Test Loss: {test_loss}')

# Вычисление MAE
mae = mean_absolute_error(Y_test1, y_pred)

print(f'Test MAE: {mae:.4f}')

plt.figure(figsize=(10, 6))
plt.plot(Y_test1[:100], label="True Values", color='blue')
plt.plot(y_pred[:100], label="Predicted Values", color='red')
plt.title('True vs Predicted values')
plt.legend()
plt.show()
