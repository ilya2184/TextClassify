import joblib

from tempfile import mkdtemp
from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import MODEL_CACHE

cachedir = mkdtemp()
memory = Memory(location=cachedir, verbose=0)

def train_model(data, model_id):
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
    joblib.dump(pipeline, f'{model_id}.joblib')

def load_model(model_id):
    if model_id in MODEL_CACHE:
        print(f"Model {model_id} loaded from cache.")
        return MODEL_CACHE[model_id]
    else:
        model = joblib.load(f'{model_id}.joblib')
        MODEL_CACHE[model_id] = model
        print(f"Model {model_id} loaded from file.")
        return model
