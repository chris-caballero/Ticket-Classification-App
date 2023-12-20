# Importing necessary libraries and modules
import html
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from utils.model_utils import *
from utils.globals import *

# Initializing Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Setting up global variables and model
model_type = 'no-pos'
tokenizer = load_tokenizer()

# Flask Application Setup
# ------------------------
# Creating Flask app instance and configuring CORS, Rate Limiting, and Caching
app = Flask(__name__, static_folder='static')
CORS(app, origins=['https://chris-caballero.github.io'])
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri='memory://'
)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
cache.set(model_type, model)

# Flask Routes
# ------------

# Model Loading Route - To load the model dynamically
@app.route('/load_model', methods=['GET'])
def load_model_route():
    try:
        global model
        if 'model' not in globals():
            model = load_model(
                model_type=model_type,
                vocab_size=len(tokenizer.vocab),
                embedding_dim=EMBEDDING_DIM,
                block_size=BLOCK_SIZE,
                num_classes=NUM_CLASSES
            )
        return jsonify({'status': 'Model loaded successfully'}), 200
    except Exception as e:
        logger.error(f"Error in /load_model: {e}")
        return jsonify({'error': 'An error occurred while loading the model.'}), 500

# Model Selection Route - For selecting different model types
@app.route('/select_model', methods=['POST'])
@cross_origin()
def select_model():
    try:
        model_type = request.form['text']

        if model_type not in ALLOWED_MODELS:
            logger.error(f"Invalid model type: {model_type}")
            return jsonify({'error': 'Invalid model type.'}), 400

        logger.info(f"Selected model: {model_type} - logger result")

        # Check if model is cached, otherwise load and cache the model
        cached_model = cache.get(model_type)
        if cached_model is None:
            # logger.debug(f"Loading and caching model: {model_type}")
            logger.info(f"Loading and caching model: {model_type}")
            model = load_model(
                model_type=model_type,
                vocab_size=len(tokenizer.vocab),
                embedding_dim=EMBEDDING_DIM,
                block_size=BLOCK_SIZE,
                num_classes=NUM_CLASSES
            )
            logger.info(f"Done Loading\n")

            cache.set(model_type, model)
        else:
            logger.info(f"Loading and caching model: {model_type}")
            # logger.debug(f"Using cached model: {model_type}")
            model = cached_model

        logger.info(f"Done with select_model. Returning jsonify...")

        return jsonify({'model': model_type})

    except Exception as e:
        logger.error(f"Error in /select_model: {e}")
        return jsonify({'error': 'An error occurred while selecting the model.'}), 500

# Text Classification Route - To classify the input text
@app.route('/classify', methods=['POST'])
@cross_origin()
def classify():
    try:
        text = request.json['text']

        logger.info(f"Classifying text: {text}")
        # sanitize user input for more robust security
        sanitized_text = html.escape(text)

        logger.info(f"Sanitized text: {sanitized_text}")
        # logger.debug(f"Classifying text: {sanitized_text}")
        topic, predicted_class = classify_text(sanitized_text, model, model_type, tokenizer, BLOCK_SIZE)
        logger.info(f"Classification result - topic: {topic}, predicted_class: {predicted_class}")

        return jsonify({'topic': topic, 'predicted_class': predicted_class})
    except Exception as e:
        logger.error(f"Error in /classify: {e}")
        return jsonify({'error': 'An error occurred while classifying the text.'}), 500

