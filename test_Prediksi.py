import joblib
import requests
import datetime
import pandas as pd
import csv
import json

# Memuat model yang sudah dilatih
model = joblib.load('rain_prediction_model.pkl')

# URL API BMKG yang valid
url = "https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=36.71.01.1003"  # Gantilah dengan URL API BMKG yang sudah kamu miliki

# Header standar untuk menghindari error 403
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Fungsi untuk mengambil data cuaca dari API BMKG
def get_weather_data():
    # Mengambil data dari API BMKG
    response = requests.get(url, headers=headers)
    
    # Memeriksa apakah request berhasil
    if response.status_code == 200:
        # Mengambil data JSON dari response
        data = response.json()
        
        # Menyiapkan data yang akan dipilih (suhu, kelembapan, kecepatan angin)
        weather_data = []
        
        for item in data["data"]:
            for cuaca in item["cuaca"]:
                for entry in cuaca:
                    # Menambahkan data cuaca yang relevan ke dalam list
                    weather_data.append({
                        "datetime": entry["datetime"],
                        "suhu": entry["t"],
                        "kelembapan": entry["hu"],
                        "kecepatan_angin": entry["ws"]
                    })
        
        return weather_data
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Fungsi untuk menghitung rata-rata per hari
def calculate_daily_averages(weather_data):
    daily_data = {}
    
    # Mengelompokkan data per hari dan menghitung rata-rata
    for day_data in weather_data:
        # Ambil tanggal tanpa waktu
        date = day_data["datetime"].split("T")[0]
        
        # Jika tanggal belum ada di daily_data, buat entry baru
        if date not in daily_data:
            daily_data[date] = {
                "suhu": [],
                "kelembapan": [],
                "kecepatan_angin": []
            }
        
        # Menambahkan nilai suhu, kelembapan, dan kecepatan angin
        daily_data[date]["suhu"].append(day_data["suhu"])
        daily_data[date]["kelembapan"].append(day_data["kelembapan"])
        daily_data[date]["kecepatan_angin"].append(day_data["kecepatan_angin"])
    
    # Menghitung rata-rata untuk setiap hari
    daily_averages = []
    for date, values in daily_data.items():
        avg_suhu = sum(values["suhu"]) / len(values["suhu"])
        avg_kelembapan = sum(values["kelembapan"]) / len(values["kelembapan"])
        avg_kecepatan_angin = sum(values["kecepatan_angin"]) / len(values["kecepatan_angin"])
        
        daily_averages.append({
            "datetime": date,
            "suhu": avg_suhu,
            "kelembapan": avg_kelembapan,
            "kecepatan_angin": avg_kecepatan_angin
        })
    
    return daily_averages

# Mengambil data cuaca dan membuat prediksi curah hujan per hari
def predict_rain():
    # Mengambil data cuaca
    weather_data = get_weather_data()
    
    if weather_data:
        # Menghitung rata-rata harian untuk suhu, kelembapan, dan kecepatan angin
        daily_averages = calculate_daily_averages(weather_data)
        
        results = []  # List untuk menyimpan hasil prediksi
        
        # Melakukan prediksi untuk setiap hari
        for day_data in daily_averages:
            # Membuat DataFrame untuk input prediksi
            input_data = pd.DataFrame([{
                "suhu": day_data["suhu"],
                "kelembapan": day_data["kelembapan"],
                "kecepatan_angin": day_data["kecepatan_angin"]
            }])
            
            # Menggunakan fitur suhu, kelembapan, dan kecepatan angin untuk prediksi
            rain_prediction = model.predict(input_data)
            print(f"Prediksi Curah Hujan pada {day_data['datetime']}: {rain_prediction[0]:.2f} mm")
            
            # Menambahkan hasil prediksi ke dalam list results
            results.append({
                "datetime": day_data["datetime"],
                "prediksi_curah_hujan": rain_prediction[0]
            })

        # Menyimpan hasil prediksi dalam format JSON
        with open("weather_prediction.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

    else:
        print("Gagal mengambil data cuaca.")

# Menjalankan fungsi untuk prediksi curah hujan
predict_rain()
