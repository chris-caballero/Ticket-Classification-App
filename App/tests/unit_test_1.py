import sys
sys.path.append('../server')

from app import app
import json

# Make sure we can access the client
def test_home():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'SupportMe' in response.data

# Sanity check that both available models can be selected
def test_select_model_valid():
    import time
    client = app.test_client()
    for model_type in ['no-pos', 'pos']:
        response = client.post('/select_model', data={'text': model_type})
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data['model'] == model_type

        time.sleep(1)

# Other models should return an error when selected
def test_select_model_invalid():
    client = app.test_client()
    response = client.post('/select_model', data={'text': 'GPT-3'})
    data = json.loads(response.data)
    assert response.status_code == 400
    assert 'error' in data

# Can we properly classify some text?
# Model does fine with missing input, and we don't allow empty inputs
# This also verifies if our preprocessing executes properly
def test_classify():
    client = app.test_client()
    response = client.post('/classify', json={'text': 'Sample text to classify.'})
    data = json.loads(response.data)
    assert response.status_code == 200
    assert 'topic' in data
    assert 'predicted_class' in data