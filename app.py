from flask import Flask, jsonify
from model_utils import predict_sp500
from flask_cors import CORS

app = Flask(__name__)

CORS(app)


@app.route("/predict", methods=["GET"])
def predict():
    try:
        print("Calling predict_sp500()...")  # debug line
        result = predict_sp500()
        print("Result:", result)  # debug line
        return jsonify(result)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)
