from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Path gambar grafik
GRAPH_PATH = 'static/sales_plot.png'

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    graph_path = None

    if request.method == 'POST':
        try:
            # Ambil input dari form
            tv = float(request.form['tv'])
            radio = float(request.form['radio'])
            newspaper = float(request.form['newspaper'])

            # Prediksi
            input_data = np.array([[tv, radio, newspaper]])
            prediction = model.predict(input_data)[0]

            # Buat grafik prediksi vs aktual
            df = pd.read_csv('advertising.csv')
            df['Prediction'] = model.predict(df[['TV', 'Radio', 'Newspaper']])

            plt.figure(figsize=(6, 4))
            plt.plot(df['Prediction'], label='Prediksi Penjualan', color='blue')
            plt.plot(df['Sales'], label='Penjualan Aktual', color='orange')
            plt.legend()
            plt.title('Prediksi vs Penjualan Aktual')
            plt.xlabel('Index')
            plt.ylabel('Penjualan')
            plt.tight_layout()

            # Simpan grafik
            plt.savefig(GRAPH_PATH)
            plt.close()

            graph_path = GRAPH_PATH

        except Exception as e:
            prediction = f"Terjadi kesalahan: {e}"

    return render_template('index.html', prediction=prediction, graph=graph_path)

if __name__ == '__main__':
    app.run(debug=True)
