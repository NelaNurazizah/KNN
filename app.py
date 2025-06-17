from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

app = Flask(__name__)

# Load model dan label encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Buat plot scatter hanya sekali
def generate_scatter_plot():
    df = pd.read_csv("Deepression.csv")
    df.columns = df.columns.str.strip()
    df = df.dropna()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="Interest",
        y="Fatigue",
        hue="Depression State",
        data=df,
        palette="viridis",
        alpha=0.8
    )
    plt.title("Interest vs Fatigue - Klasifikasi KNN")
    plt.xlabel("Interest")
    plt.ylabel("Fatigue")
    plt.legend(title="Depression State")
    plt.tight_layout()
    plt.savefig("static/images/knn_scatter.png")
    plt.close()

# Jalankan saat server mulai
generate_scatter_plot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form.get("Sleep")),
            int(request.form.get("Appetite")),
            int(request.form.get("Interest")),
            int(request.form.get("Fatigue")),
            int(request.form.get("Worthlessness")),
            int(request.form.get("Concentration")),
            int(request.form.get("Agitation")),
            int(request.form.get("Suicidal_Ideation")),
            int(request.form.get("Sleep_Disturbance")),
            int(request.form.get("Aggression")),
            int(request.form.get("Panic_Attacks")),
            int(request.form.get("Hopelessness")),
            int(request.form.get("Restlessness")),
            int(request.form.get("Low_Energy"))
        ]

        prediction_encoded = model.predict([features])[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        return render_template("result.html", prediction=prediction_label, image_path="images/knn_scatter.png")
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
