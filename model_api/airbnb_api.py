from flask import Flask, jsonify, request
from api_utils import (
    process_features_airbnb, get_model_features,
    load_model, make_predictions_airbnb
)

app = Flask(__name__)


ARTIFACT_PATH = './models/simple_classifier.pkl'
# Important: this list must follow the order used by the DS in training
FEATURES_TO_GET = [
    'neighbourhood', 'room_type', 'accommodates', 'bathrooms', 'bedrooms'
]
ESTIMATOR = load_model(path=ARTIFACT_PATH)


@app.route('/', methods=['GET'])
def home():
    return '<h1>airbnb API home page</h1>'


@app.route('/predict/', methods=['POST', 'GET'])
def make_predictions():
    features = get_model_features(
        request.values,
        model_features=FEATURES_TO_GET
    )
    features_proc = process_features_airbnb(features=features)
    prediction = make_predictions_airbnb(
        model=ESTIMATOR,
        features=features_proc
    )
    return jsonify({
        'id': request.values['id'],
        'price_category': prediction
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
