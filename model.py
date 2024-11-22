import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import mean_squared_error

# Загрузка предобработанных данных
X_train = pd.read_csv("X_train.csv").values  # Загрузка данных
X_test = pd.read_csv("X_test.csv").values
y_train = pd.read_csv("y_train.csv").values.flatten()
y_test = pd.read_csv("y_test.csv").values.flatten()

# Преобразование данных для LSTM (добавляем временную ось)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # (samples, timesteps, features)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Параметры модели
input_shape = (X_train.shape[1], X_train.shape[2])  # (временные шаги, количество признаков)

# Создание модели с BiLSTM
model = Sequential([
    Input(shape=input_shape),  # Входной слой
    Bidirectional(LSTM(64, return_sequences=True, activation='relu')),  # BiLSTM слой с возвратом последовательностей
    Dropout(0.2),                                                 # Dropout для предотвращения переобучения
    Bidirectional(LSTM(32, activation='relu')),                   # BiLSTM слой без возврата последовательностей
    Dropout(0.2),
    Dense(32, activation='relu'),                                 # Полносвязный слой
    Dense(1)                                                      # Выходной слой
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Вывод структуры модели
print("Структура модели:")
model.summary()

# EarlyStopping для предотвращения переобучения
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10, #Если val_loss не уменьшается в течение 10 последовательных эпох, обучение будет остановлено.
    restore_best_weights=True, #После остановки обучения веса модели будут восстановлены до значений, которые были в эпоху с лучшей (минимальной) val_loss.
    verbose=1
)

# Обучение модели
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# Оценка модели
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Предсказания и расчет метрики RMSE
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f"Test RMSE: {rmse:.4f}")

# Сохранение модели
model.save("bilstm_model.h5")

# Визуализация обучения
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
