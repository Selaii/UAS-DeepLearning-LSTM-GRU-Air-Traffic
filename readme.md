📈 Passenger Forecasting using LSTM and GRU
Proyek ini bertujuan untuk memprediksi jumlah penumpang penerbangan per bulan menggunakan metode Deep Learning berbasis LSTM dan GRU. Model diuji pada berbagai ukuran jendela waktu (window size) untuk membandingkan performa.

📁 Dataset
Dataset yang digunakan adalah statistik bulanan penumpang penerbangan domestik dan internasional.

Nama file: air traffic.csv

Jumlah baris: ±180 baris (data bulanan dari 2009–2023)

Kolom utama: Month, Dom_Pax, Int_Pax, Pax, dll.

🔗 Dataset Lengkap
https://www.kaggle.com/datasets/yyxian/u-s-airline-traffic-data?resource=download

🛠️ Dependencies
Pastikan Python versi 3.8–3.11 sudah terpasang, lalu instal dependensi berikut:

📦 Install via pip
pip install numpy pandas matplotlib scikit-learn tensorflow

Atau, gunakan
pip install -r requirements.txt

📄 Struktur File
📁 Project/
├── LSTM_Model.py         # Skrip prediksi dengan LSTM
├── GRU_Model.py          # Skrip prediksi dengan GRU
├── air traffic.csv       # Dataset jumlah penumpang
├── Hasil.xlsx            # Hasil Membandingkan MAPE dari LSTM dan GRU
├── README.md             # Dokumentasi proyek
├── requirements.txt      # Yang perlu di install

🧪 Model Evaluasi
Model dinilai menggunakan 3 metrik utama:

RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
MAPE (Mean Absolute Percentage Error)

▶️ Cara Menjalankan
1. Pastikan file air traffic.csv berada di folder yang sama dengan file .py
2. Install Dependencies menggunakan
pip install -r requirements.txt

3. Jalankan file LSTM:
python LSTM_Model.py

4. Jalankan file GRU:
python GRU_Model.py

📊 Hasil Singkat
Model diuji dengan variasi window_size dari 6 hingga 21. Hasil menunjukkan bahwa GRU dengan window_size = 15 menghasilkan performa terbaik berdasarkan MAPE (% kesalahan prediksi rata-rata terendah).