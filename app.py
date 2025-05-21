from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
from datetime import datetime
import csv
import os

app = Flask(__name__)

# Load model dan scaler
reg_model = joblib.load("models/rf_regression_model.pkl")
clf_model = joblib.load("models/knn_classifier_model.pkl")
scaler = joblib.load("models/scaler_knn.pkl")

# Mapping posisi pemain
position_mapping = {
    "ST": 0, "CF": 1, "CAM": 2, "CM": 3, "CDM": 4, "LM": 5, "RM": 6,
    "LW": 7, "RW": 8, "CB": 9, "LB": 10, "RB": 11, "LWB": 12, "RWB": 13, "GK": 14
}

features = ['overall', 'potential', 'age', 'wage_eur', 'height_cm',
            'weight_kg', 'player_positions', 'skill_moves', 'weak_foot', 'international_reputation',
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']

@app.route("/")
def redirect_to_welcome():
    return redirect(url_for("welcome"))

@app.route("/welcome")
def welcome():
    return render_template("welcome.html")

@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Ambil input dari form
            data = []
            for f in features:
                if f == 'player_positions':
                    pos = request.form[f]
                    encoded = position_mapping.get(pos, 0)
                    data.append(encoded)
                else:
                    data.append(float(request.form[f]))

            df = pd.DataFrame([data], columns=features)

            # Prediksi
            price = reg_model.predict(df)[0]
            scaled = scaler.transform(df)
            category = clf_model.predict(scaled)[0]

            # 🔒 Logging ke CSV
            log_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **{f: request.form[f] for f in features},
                "prediction_eur": round(price),
                "kategori": category
            }

            log_file = "log.csv"
            file_exists = os.path.isfile(log_file)

            with open(log_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=log_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_data)

            return render_template("index.html", prediction=round(price), kategori=category)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
