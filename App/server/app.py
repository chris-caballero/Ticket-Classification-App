import torch
import requests
from transformers import AutoTokenizer
from preprocessing_util import *
from models.model_schema.model import EncoderTransformer
from torch import from_numpy, tensor
from torch.nn.functional import softmax
from flask import Flask, request, jsonify, send_from_directory

# HYPER-PARAMETERS

topics = ['Bank Account Services', 'Credit card / Prepaid card', 'Others', 'Theft / Dispute reporting', 'Mortgages / Loans']
BLOCK_SIZE = 200
NUM_CLASSES = 5
EMBEDDING_DIM = 300

model_type = 'no-pos'
MODEL_PATH = f'models/trained_models/text_classification_{model_type}.pth'
model = None
tokenizer = None

app = Flask(__name__, static_folder='../client')

@app.route('/')
def serve_index():
    return send_from_directory('../client', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../client', filename)

@app.route('/select_model', methods=['POST'])
def select_model():
    model_type = request.form['text']

    print(model_type)
    MODEL_PATH = f'models/trained_models/text_classification_{model_type}.pth'
    model = load_model(MODEL_PATH)

    return jsonify({'model': model_type})

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    topic, predicted_class = classify_text(text)
    return jsonify({'topic': topic, 'predicted_class': predicted_class})

def classify_text(text):
    input_ids = preprocess_and_encode_text(text)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    return topics[predicted_class], predicted_class

def preprocess_and_encode_text(text, model_type='no-pos'):
    # Apply necessary preprocessing based on model type, 'route' model in future
    text = preprocessing_fn(text, model_type=model_type)

    # Tokenize the text and create input_ids
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=BLOCK_SIZE,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )

    return encoding['input_ids'][0].unsqueeze(0)

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

def load_model(path):
    model = EncoderTransformer(len(tokenizer.vocab), EMBEDDING_DIM, BLOCK_SIZE, NUM_CLASSES)
    model.load_state_dict(torch.load(path))
    return model

if __name__ == '__main__':
    tokenizer = load_tokenizer()
    model = load_model(MODEL_PATH)
    model.eval()
    app.run()
