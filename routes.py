import pandas as pd

from flask import request, jsonify
from model import train_model, test_model, predict_model

def setup_routes(app):
    @app.route('/train', methods=['POST'])
    def train():
        data = pd.read_excel(request.files['file'])
        model_id = request.form['model_id']
        train_model(data, model_id)
        return jsonify({"message": "Model trained successfully."})

    @app.route('/test', methods=['POST'])
    def test():
        data = pd.read_excel(request.files['file'])
        model_id = request.form['model_id']
        accuracy = test_model(data, model_id)
        return jsonify({"accuracy": accuracy})

    @app.route('/predict', methods=['POST'])
    def predict():
        model_id = request.form['model_id']
        operation = request.form['operation']
        text = request.form['text']
        data = pd.DataFrame({ 'ВидОперации': [operation], 'НазначениеПлатежа': [text] })
        predata = predict_model(data, model_id)
        return jsonify({
            "article": predata.get("prediction"),
            "confidence": predata.get("confidence")
        })