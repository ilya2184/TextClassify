
from flask import request, jsonify
from model import train_model, load_model
import pandas as pd

def setup_routes(app):
    @app.route('/train', methods=['POST'])
    def train():
        data = pd.read_excel(request.files['file'])
        model_id = request.form['model_id']
        train_model(data, model_id)
        return jsonify({"message": "Model trained successfully."})

    @app.route('/predict', methods=['POST'])
    def predict():
        model_id = request.form['model_id']
        operаtaion = request.form['operаtaion']
        text = request.form['text']
        sample_data = pd.DataFrame({ 'ВидОперации': [operаtaion], 'НазначениеПлатежа': [text] })
        model = load_model(model_id)
        prediction = model.predict(sample_data)
        return jsonify({"article": prediction[0]})
