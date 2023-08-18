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

# Set default model type and load tokenizer and model
model_type = 'no-pos'
tokenizer = load_tokenizer()
model = load_model(
    model_type=model_type,
    vocab_size=len(tokenizer.vocab),
    embedding_dim=EMBEDDING_DIM,
    block_size=BLOCK_SIZE,
    num_classes=NUM_CLASSES
)

# Create a Flask app instance
app = Flask(__name__, static_folder='../client')

# Set up rate limiting using Flask Limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri='memory://'
)

# Set up caching using Flask Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Route to serve the index.html file
@app.route('/')
def serve_index():
    return send_from_directory('../client', 'index.html')

# Route to serve static files (CSS, JavaScript, etc.)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../client', filename)

# Route to select the model type
@app.route('/select_model', methods=['POST'])
@limiter.limit("1/second")
def select_model():
    try:
        model_type = request.form['text']

        if model_type not in ALLOWED_MODELS:
            return jsonify({'error': 'Invalid model type.'}), 400

        logger.info(f"Selected model: {model_type}")

        # Check if model is cached, otherwise load and cache the model
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

# Route to classify text
@app.route('/classify', methods=['POST'])
@limiter.limit("1/second")
def classify():
    try:
        text = request.json['text']
        # santitize user input for more robust security
        sanitized_text = html.escape(text)
        topic, predicted_class = classify_text(sanitized_text, model, model_type, tokenizer, BLOCK_SIZE)
        return jsonify({'topic': topic, 'predicted_class': predicted_class})
    except Exception as e:
        logger.error(f"Error in /classify: {e}")
        return jsonify({'error': 'An error occurred while classifying the text.'}), 500

# Run the app if executed as the main script
if __name__ == '__main__':
    app.run()
