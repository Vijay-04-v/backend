from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # <-- This line enables cross-origin requests

# Load model, scaler and encoder
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']
    input_data = np.array(data).reshape(1, -1)
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    label = label_encoder.inverse_transform(prediction)
    return jsonify({'prediction': label[0]})

if __name__ == '__main__':
    app.run(debug=True)
