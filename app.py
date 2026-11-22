
import streamlit as st
import pandas as pd
import joblib

# Memuat model AI
model = joblib.load("model_prediksi_cpu.pkl")

st.set_page_config(
    page_title="Prediksi Performa Komputer",
    page_icon="💻"
)

st.title("💻 Prediksi Performa Komputer")

st.write(
    "Prediksi performa relatif CPU menggunakan "
    "Jaringan Syaraf Tiruan Multilayer Perceptron."
)

st.subheader("Masukkan Spesifikasi Hardware")

col1, col2 = st.columns(2)

with col1:
    myct = st.number_input("MYCT", min_value=1, value=125)
    mmin = st.number_input("MMIN", min_value=0, value=256)
    mmax = st.number_input("MMAX", min_value=0, value=6000)

with col2:
    cach = st.number_input("CACH", min_value=0, value=256)
    chmin = st.number_input("CHMIN", min_value=0, value=16)
    chmax = st.number_input("CHMAX", min_value=0, value=128)

if st.button("Prediksi Performa"):
    data = pd.DataFrame([{
        "MYCT": myct,
        "MMIN": mmin,
        "MMAX": mmax,
        "CACH": cach,
        "CHMIN": chmin,
        "CHMAX": chmax
    }])

    hasil = model.predict(data)[0]

    st.success("Prediksi berhasil!")
    st.metric(
        "Prediksi Performa Relatif CPU (PRP)",
        f"{hasil:.2f}"
    )

    st.caption(
        "Hasil merupakan estimasi model berdasarkan "
        "dataset Computer Hardware UCI."
    )
