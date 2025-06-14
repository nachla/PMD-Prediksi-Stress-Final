import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import joblib

# ------------------ Load Models ------------------
@st.cache_resource
def load_models():
    svm = joblib.load("svm_model.pkl")
    rf = joblib.load("rf_model.pkl")
    knn = joblib.load("knn_model.pkl")
    return svm, rf, knn

svm_model_data, rf_model, knn_model = load_models()
gamma = 0.1

# ------------------ Fungsi SVM Manual ------------------
def rbf_kernel(X1, X2, gamma=0.1):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            diff = X1[i] - X2[j]
            K[i, j] = np.exp(-gamma * np.dot(diff, diff))
    return K

def predict_ovr_svm(X_test, models, gamma=0.1):
    scores = []
    for cls, model in models.items():
        alpha = model['alpha']
        sv = model['support_vectors']
        sv_labels = model['support_vector_labels']
        score = np.dot(rbf_kernel(X_test, sv, gamma), alpha * sv_labels)
        scores.append(score)
    scores = np.array(scores)
    predictions = np.argmax(scores, axis=0)
    class_labels = list(models.keys())
    return np.array([class_labels[i] for i in predictions])

label_mapping = {
    0: "Tingkat Stres Rendah, sepertinya Anda Mahasiswa yang mulai merasa lelah karena tugas kuliah atau tekanan sosial ringan.",
    1: "Tingkat Stres Sedang, sepertinya Anda Mahasiswa yang mengalami tekanan akademik + masalah pribadi ringan.",
    2: "Tingkat Stres Tinggi, sepertinya Anda Mahasiswa yang mengalami burnout, depresi ringan–sedang, atau tekanan sosial besar."
}

# ------------------ Sidebar NAVIGATION ------------------
with st.sidebar:
    selected = option_menu(
        "Menu", 
        ["Deskripsi", "Klasifikasi"], 
        icons=["info-circle", "activity"], 
        menu_icon="cast", 
        default_index=0,
        orientation="vertical"
    )

# ------------------ Halaman DESKRIPSI ------------------
if selected == "Deskripsi":
    st.title("Tentang Sistem Deteksi Stres Mahasiswa")

    st.subheader("📊 Fitur Input yang Digunakan:")
    st.markdown("""
    - **Self Esteem**: Tingkat kepercayaan diri mahasiswa (0–10)
    - **Depression**: Tingkat depresi yang dirasakan (0–10)
    - **Anxiety Level**: Tingkat kecemasan atau gelisah (0–10)
    - **Sleep Quality**: Kualitas tidur (0–10)
    - **Bullying**: Tingkat pengalaman atau paparan bullying (0–10)
    """)

    st.subheader("🧠 Kategori Tingkat Stres:")
    st.markdown("""
    - `0` → **Tingkat Stres Rendah**  
    - `1` → **Tingkat Stres Sedang**  
    - `2` → **Tingkat Stres Tinggi**
    """)

    st.subheader("📊 Saran Penanganan:")
    st.markdown("""
    1. Stres Rendah

Ciri-ciri: Merasa lelah, sedikit cemas, tetapi masih bisa menjalankan aktivitas harian dengan baik.

Saran Penanganan:
- Manajemen waktu ringan: Gunakan to-do list dan teknik pomodoro.
- Aktivitas relaksasi ringan: Mendengarkan musik, menggambar, menonton film ringan.
- Olahraga ringan: Jalan kaki 20 menit, stretching, yoga ringan.
- Tidur teratur: 7–8 jam/hari dengan jadwal tidur konsisten.
- Jurnal syukur: Menulis 3 hal positif setiap hari.
    
    2. Stres Sedang
                
Ciri-ciri: Sulit fokus, mulai sering menunda tugas, perubahan pola tidur/makan, mudah marah.

Saran Penanganan:
- Teknik relaksasi: Pernapasan dalam (deep breathing), meditasi (apps: Headspace, Calm).
- Manajemen stres: Teknik CBT sederhana, menulis jurnal emosi.
- Sosialisasi: Ngobrol dengan teman atau keluarga terdekat.
- Manajemen beban kerja: Prioritaskan tugas, jangan multitasking.
- Konsultasi ringan: Bicarakan ke dosen pembimbing atau layanan konseling kampus.

    3. Stres Tinggi
                
Ciri-ciri: Gangguan tidur berat, kehilangan motivasi, gejala fisik (sakit kepala, mual), merasa putus asa.

Saran Penanganan:
- Konseling profesional: Psikolog kampus, konselor berlisensi, layanan telekonseling (seperti Sehat Jiwa, Riliv).
- Pendampingan rutin: Monitoring suasana hati, aplikasi mental health (Mindtera, Riliv, MindfulnessID).
- Intervensi gaya hidup: Pola makan sehat, kurangi kafein/gadget, olahraga intensitas sedang.
- Rencana darurat (jika depresi berat): Hubungi layanan darurat psikologis (contoh: Satgas Kesehatan Jiwa, call center kampus).
- Terapi jangka panjang (jika kronis): CBT atau terapi interpersonal.
    """)

    st.info("Gunakan menu 'Klasifikasi' untuk melakukan prediksi.")

# ------------------ Halaman KLASIFIKASI ------------------
elif selected == "Klasifikasi":
    st.title("Deteksi Tingkat Stres Mahasiswa")
    st.write("Masukkan nilai-nilai berikut (0–10):")

    self_esteem = st.slider("Self Esteem", 0.0, 10.0, 5.0)
    depression = st.slider("Depression", 0.0, 10.0, 5.0)
    anxiety = st.slider("Anxiety Level", 0.0, 10.0, 5.0)
    sleep_quality = st.slider("Sleep Quality", 0.0, 10.0, 5.0)
    bullying = st.slider("Bullying", 0.0, 10.0, 5.0)

    features = np.array([[self_esteem, depression, anxiety, sleep_quality, bullying]])
    model_choice = st.selectbox("Pilih Model Klasifikasi", ["SVM", "Random Forest", "K-Nearest Neighbors"])


    if st.button("Prediksi Tingkat Stres"):
        if model_choice == "SVM":
            prediction = predict_ovr_svm(features, svm_model_data, gamma=gamma)
        elif model_choice == "Random Forest":
            prediction = rf_model.predict(features)
        elif model_choice == "K-Nearest Neighbors":
            prediction = knn_model.predict(features)
        else:
            prediction = ["Model tidak dikenali"]

        label = prediction[0]
        label_desc = label_mapping.get(label, "Label tidak dikenali")
        st.success(f"Tingkat Stres yang Diprediksi: {label} ({label_desc})")
