import os
import joblib

from tempfile import mkdtemp
from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from config import MODEL_CACHE, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)

cachedir = mkdtemp()
memory = Memory(location=cachedir, verbose=0)

def train_model(data, model_id):
    data = data[['ВидОперации', 'НазначениеПлатежа', 'СтатьяДвиженияДенежныхСредств']].dropna()
    X = data[['ВидОперации', 'НазначениеПлатежа']]
    y = data['СтатьяДвиженияДенежныхСредств']
    column_transformer = ColumnTransformer([
        ('vid_operacii_tfidf', TfidfVectorizer(), 'ВидОперации'),
        ('naznach_platega_tfidf', TfidfVectorizer(), 'НазначениеПлатежа')
    ])
    pipeline = Pipeline([
        ('transformer', column_transformer),
        ('clf', LogisticRegression(random_state=42))
    ], memory=memory)
    
    pipeline.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f'model_{model_id}.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

def test_model(data, model_id):
    data = data[['ВидОперации', 'НазначениеПлатежа', 'СтатьяДвиженияДенежныхСредств']].dropna()
    X = data[['ВидОперации', 'НазначениеПлатежа']]
    y = data['СтатьяДвиженияДенежныхСредств']
    model = load_model(model_id)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy

def load_model(model_id):
    if model_id in MODEL_CACHE:
        print(f"Model {model_id} loaded from cache.")
        return MODEL_CACHE[model_id]
    else:
        model_path = os.path.join(MODEL_DIR, f'model_{model_id}.joblib')
        model = joblib.load(model_path)
        MODEL_CACHE[model_id] = model
        print(f"Model {model_id} loaded from file.")
        return model
