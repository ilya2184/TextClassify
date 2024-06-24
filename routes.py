import pandas as pd

from flask import request, jsonify
from model import train_model, test_model, predict_model
from textutils import find_best_match

def setup_routes(app):

    @app.route('/ping', methods=['GET','POST'])
    def ping():
        return jsonify({
            "Ready": "POST train, test, predict or stringmatch. https://github.com/ilya2184/TextClassify"
        })

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
    
    @app.route('/stringmatch', methods=['POST'])
    def stringmatch():
        data = request.get_json()
        strings_list = data.get('string_list', [])
        text = data.get('text', '')
        prediction = find_best_match(strings_list, text)
        return jsonify({
            "prediction": prediction
        })
   