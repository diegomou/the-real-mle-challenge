from flask import Flask, jsonify, request
from api_utils import (
    airbnb_api_features_processing, get_model_features,
    load_model, make_model_prediction
)

app  = Flask(__name__)


ARTIFACT_PATH = './models/simple_classifier.pkl'
# Important: this list must follow the order used by the DS when model was trained
FEATURES_TO_GET = [
    'neighbourhood', 'room_type', 'accommodates', 'bathrooms', 'bedrooms'
]
ESTIMATOR = load_model(path=ARTIFACT_PATH)


@app.route('/predict/', methods=['POST'])
def make_predictions():
    if request.method == 'POST':
        features = get_model_features(request.values, model_features=FEATURES_TO_GET)
        features_proc = airbnb_api_features_processing(features)
        prediction = make_model_prediction(model=ESTIMATOR, features=features_proc)
        return jsonify({
            'id': request.values['id'],
            'prediction': prediction
        })
