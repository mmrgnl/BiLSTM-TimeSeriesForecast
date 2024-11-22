import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Загрузка данных
training_data = pd.read_csv("power-laws-forecasting-energy-consumption-training-data.csv", delimiter=';')

# Убедимся, что Timestamp преобразован в datetime
if 'Timestamp' in training_data.columns:
    training_data['Timestamp'] = pd.to_datetime(training_data['Timestamp'])

# Применяем логарифмическое преобразование для столбца 'Value', добавляем 1 для избегания логарифма от 0
training_data['LogValue'] = np.log1p(training_data['Value'])

# Визуализация 1: Распределение логарифмированного потребления (LogValue)
plt.figure(figsize=(10, 6))
sns.histplot(training_data['LogValue'], bins=50, kde=True)
plt.title("Распределение логарифмированного потребления энергии", fontsize=16)
plt.xlabel("Логарифм потребления энергии (LogValue)")
plt.ylabel("Частота")
plt.show()

# Визуализация 2: Временной ряд для первого здания (SiteId) с логарифмированными значениями
site_id = training_data['SiteId'].iloc[0]  # Выберем первое здание
site_data = training_data[training_data['SiteId'] == site_id]

plt.figure(figsize=(14, 6))
plt.plot(site_data['Timestamp'], site_data['LogValue'], label=f"SiteId: {site_id}")
plt.title("Временной ряд логарифмированного потребления энергии", fontsize=16)
plt.xlabel("Время")
plt.ylabel("Логарифм потребления энергии")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


