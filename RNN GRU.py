# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# 2. Load dataset
df = pd.read_csv('air traffic.csv', parse_dates=['Month'])
df.columns = df.columns.str.strip()  # Hilangkan spasi ekstra
print("Kolom-kolom tersedia:", df.columns)

# 3. Pastikan kolom target tersedia
if 'Pax' not in df.columns:
    raise ValueError("Kolom 'Pax' tidak ditemukan. Cek nama kolom CSV.")

# 4. Urutkan dan set index waktu
df = df.sort_values('Month')
df = df.set_index('Month')

# 5. Pastikan kolom Pax numerik (hilangkan koma dan ubah ke float)
df['Pax'] = df['Pax'].replace(',', '', regex=True).astype(float)

# 6. Ambil data target
data = df[['Pax']].values

# 7. Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 8. Buat window sequence
def create_sequences(data, window_size=12):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window_size = 12
X, y = create_sequences(data_scaled, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))  # reshape untuk GRU

# 9. Split train-test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 10. Buat model GRU
model = Sequential([
    GRU(64, input_shape=(window_size, 1), return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 11. Latih model
history = model.fit(
    X_train, y_train,
    epochs=50, batch_size=16,
    validation_data=(X_test, y_test),
    verbose=2
)

# 12. Evaluasi
preds = model.predict(X_test)
preds_inv = scaler.inverse_transform(preds)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Hitung RMSE, MAE, MAPE
rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
mae = mean_absolute_error(y_test_inv, preds_inv)
mape = np.mean(np.abs((y_test_inv - preds_inv) / y_test_inv)) * 100

# Hitung versi persen berdasarkan rata-rata nilai aktual
mean_actual = np.mean(y_test_inv)
rmse_percent = (rmse / mean_actual) * 100
mae_percent = (mae / mean_actual) * 100

# Cetak hasil error
print(f"Error Metrics:")
print(f"RMSE: {rmse:,.2f} passengers ({rmse_percent:.2f}%)")
print(f"MAE : {mae:,.2f} passengers ({mae_percent:.2f}%)")
print(f"MAPE: {mape:.2f}%")

# 13. Visualisasi hasil prediksi
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size + window_size:], y_test_inv, label='Actual')
plt.plot(df.index[train_size + window_size:], preds_inv, label='Predicted')
plt.title('Passenger Prediction vs Actual (GRU)')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.tight_layout()
plt.show()

# 14. Plot loss training vs val
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Loss per Epoch (GRU)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.show()
