from flask import Flask, request, jsonify
import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("winners_f1_1950_2025_v2.csv")

# Limpiar nombres de columnas
df.columns = [col.strip().lower().replace(" ", "_").replace("\\", "") for col in df.columns]

# Inicializar la app Flask
app = Flask(__name__)

@app.route("/search", methods=["GET"])
def search():
    year = request.args.get("year")
    grand_prix = request.args.get("grand_prix")
    driver = request.args.get("driver")

    results = df.copy()

    if year:
        results = results[results["year"] == int(year)]
    if grand_prix:
        results = results[results["grand_prix"].str.contains(grand_prix, case=False, na=False)]
    if driver:
        results = results[results["winner_name"].str.contains(driver, case=False, na=False)]

    output = results[["date", "grand_prix", "circuit", "winner_name", "team", "time", "laps", "year"]].to_dict(orient="records")
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
