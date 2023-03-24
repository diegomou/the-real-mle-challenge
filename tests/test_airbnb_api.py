from model_api.airbnb_api import make_predictions


def test_home_page():
    flask_app = make_predictions()

    # Create a test client using the Flask application configured for testing
    with flask_app.test_client() as test_client:
        response = test_client.post('/predict?id=1&accommodates=2&room_type=Entire home/apt&beds=2&bedrooms=1&bathrooms=2&neighbourhood=Brooklyn&tv=1&elevator=1&internet=0&latitude=40.71&longitude=-73.96')
        assert response.status_code == 200
        assert b"Welcome to the" in response.data
        assert b"Flask User Management Example!" in response.data
        assert b"Need an account?" in response.data
        assert b"Existing user?" in response.data