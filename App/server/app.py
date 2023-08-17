import html
import logging

from flask import Flask, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache

from utils.model_utils import *
from utils.globals import *

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


model_type='no-pos'
tokenizer = load_tokenizer()
model = load_model(
    model_type=model_type, 
    vocab_size=len(tokenizer.vocab), 
    embedding_dim=EMBEDDING_DIM, 
    block_size=BLOCK_SIZE, 
    num_classes=NUM_CLASSES
)

app = Flask(__name__, static_folder='../client')
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri='memory://'
)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/')
def serve_index():
    return send_from_directory('../client', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../client', filename)

@app.route('/select_model', methods=['POST'])
@limiter.limit("1/second")
def select_model():
    try:
        model_type = request.form['text']

        if model_type not in ALLOWED_MODELS:
            return jsonify({'error': 'Invalid model type.'}), 400

        logger.info(f"Selected model: {model_type}")

        cached_model = cache.get(model_type)
        if cached_model is None:
            model = load_model(
                model_type=model_type, 
                vocab_size=len(tokenizer.vocab), 
                embedding_dim=EMBEDDING_DIM, 
                block_size=BLOCK_SIZE, 
                num_classes=NUM_CLASSES
            )
            cache.set(model_type, model)
        else:
            model = cached_model

        return jsonify({'model': model_type})

    except Exception as e:
        logger.error(f"Error in /select_model: {e}")
        return jsonify({'error': 'An error occurred while selecting the model.'}), 500

@app.route('/classify', methods=['POST'])
@limiter.limit("1/second")
def classify():
    try:
        text = request.json['text']
        sanitized_text = html.escape(text)
        topic, predicted_class = classify_text(sanitized_text, model, model_type, tokenizer, BLOCK_SIZE)
        return jsonify({'topic': topic, 'predicted_class': predicted_class})
    except Exception as e:
        logger.error(f"Error in /classify: {e}")
        return jsonify({'error': 'An error occurred while classifying the text.'}), 500
    
if __name__ == '__main__':
    app.run()
