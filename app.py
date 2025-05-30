import pickle
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import os

# Muat model yang telah dilatih
try:
    model_dict = pickle.load(open('./hh.p', 'rb'))
    model = model_dict['model']
    print("Model 'hh.p' berhasil dimuat.")
except FileNotFoundError:
    print("Error: File model 'hh.p' tidak ditemukan. Pastikan model berada di direktori yang sama.")
    exit() # Keluar jika model tidak ditemukan
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# Inisialisasi aplikasi Flask
app = Flask(__name__)
# Konfigurasi kunci rahasia untuk Flask-SocketIO (penting untuk keamanan)
app.config['SECRET_KEY'] = 'your_secret_key_here' # Ganti dengan kunci rahasia yang kuat!
socketio = SocketIO(app, cors_allowed_origins="*") # Izinkan semua origin untuk pengembangan

# Kamus untuk memetakan label numerik ke huruf Hijaiyah dan Latin
labels_dict = {
    0: 'Alif / ﺍ', 1: 'Ba / ﺏ', 2: 'Ta / ﺕ', 3: 'Tsa / ﺙ', 4: 'Jim / ﺝ',
    5: 'Ha / ﺡ', 6: 'Kho / ﺥ', 7: 'Dal / ﺩ', 8: 'Dzal / ﺫ', 9: 'Ra / ﺭ',
    10: 'Zay / ﺯ', 11: 'Sin / ﺱ', 12: 'Syin / ﺵ', 13: 'Shod / ﺹ', 14: 'Dhod / ﺽ',
    15: 'Tho / ﻁ', 16: 'Zho / ﻅ', 17: 'Ain / ﻉ', 18: 'Ghain / ﻍ', 19: 'Fa / ﻑ',
    20: 'Kaf / ﻙ', 21: 'Qof / ﻕ', 22: 'Lam / ﻝ', 23: 'Mim / ﻡ', 24: 'Nun / ﻥ',
    25: 'Waw / ﻭ', 26: 'Ha\' / ﻩ', 27: 'Ya / ﻱ'
}

CONFIDENCE_THRESHOLD = 0.30 # Ambang batas kepercayaan untuk prediksi

@app.route('/')
def index():
    """Merender halaman HTML utama."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Menangani koneksi baru dari klien Socket.IO."""
    print('Klien terhubung:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Menangani pemutusan koneksi dari klien Socket.IO."""
    print('Klien terputus:', request.sid)

@socketio.on('process_landmarks')
def handle_process_landmarks(data_aux):
    """
    Menerima data landmark dari frontend, melakukan prediksi,
    dan mengirim hasilnya kembali ke klien.
    """
    try:
        # Konversi data_aux (list dari JS) ke array NumPy
        features = np.asarray(data_aux).reshape(1, -1) # Pastikan bentuknya (1, jumlah_fitur)

        # Lakukan prediksi probabilitas menggunakan model
        probs = model.predict_proba(features)[0]
        max_prob = np.max(probs)
        predicted_class = np.argmax(probs)

        predicted_label = "Tidak Diketahui"
        arabic_label = ""
        accuracy_text = "Akurasi: -"

        if max_prob >= CONFIDENCE_THRESHOLD:
            # Jika probabilitas di atas ambang batas, tampilkan label
            label_full = labels_dict.get(predicted_class, "Tidak Diketahui")
            # Pisahkan label Latin dan Arab
            parts = label_full.split('/')
            predicted_label = parts[0].strip()
            arabic_label = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Jika di bawah ambang batas, set sebagai "Tidak Diketahui"
            predicted_label = "Tidak Diketahui"
            arabic_label = ""

        accuracy_text = f"Akurasi: {max_prob * 100:.2f}%"

        # Kirim hasil prediksi kembali ke klien
        emit('prediction_result', {
            'predicted_label': predicted_label,
            'arabic_label': arabic_label,
            'accuracy_text': accuracy_text
        })
    except Exception as e:
        print(f"Error saat memproses landmark: {e}")
        # Kirim pesan error ke klien jika diperlukan
        emit('prediction_result', {
            'predicted_label': "Error",
            'arabic_label': "",
            'accuracy_text': f"Error: {e}"
        })

if __name__ == '__main__':
    from flask import request # Import di sini untuk menghindari circular import
    # Jalankan aplikasi Flask-SocketIO
    # host='0.0.0.0' agar dapat diakses dari perangkat lain di jaringan yang sama
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
