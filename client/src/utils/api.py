from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route("/predict")
def predict():
    letras = ["A", "B", "C", "HOLA", "GRACIAS"]
    return jsonify({"prediccion": random.choice(letras)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)