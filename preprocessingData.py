import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data_file = "power-laws-forecasting-energy-consumption-training-data.csv"
training_data = pd.read_csv(data_file, delimiter=';')


training_data = training_data.dropna()

training_data = training_data.drop_duplicates()

training_data = training_data[training_data['Value'] > 0]
print(f"Размер Value : {training_data.shape}")

training_data = training_data[training_data['Value'] <= 100000]

print(f"Размер Value > 100000: {training_data.shape}")


data = training_data

# Преобразование столбца Timestamp
data["Timestamp"] = pd.to_datetime(data["Timestamp"], utc=True)

# Извлечение полезных признаков
data["year"] = data["Timestamp"].dt.year
data["month"] = data["Timestamp"].dt.month
data["day"] = data["Timestamp"].dt.day
data["hour"] = data["Timestamp"].dt.hour
data["minute"] = data["Timestamp"].dt.minute
data["day_of_week"] = data["Timestamp"].dt.weekday
data["day_of_year"] = data["Timestamp"].dt.dayofyear

data = data.drop(columns=["Timestamp"])

data = data.apply(pd.to_numeric, errors='coerce')  # Преобразование всех значений в числовые
data = data.dropna()  # Удаляем строки с NaN

features = data.drop(columns=["Value"])
target = data["Value"]

# Масштабирование числовых признаков
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42
)
y_train = y_train/ 1_000
y_test = y_test/ 1_000
# Сохранение предобработанных данных
pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

print("Предобработка данных завершена.")
print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
