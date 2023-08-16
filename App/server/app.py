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

MODEL_PATH = 'models/trained_models/text_classification_no-pos.pth'
model, tokenizer = None, None

app = Flask(__name__, static_folder='../client')

@app.route('/')
def serve_index():
    return send_from_directory('../client', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../client', filename)

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

def preprocess_and_encode_text(text):
    text = preprocessing_fn(text)

    # Tokenize the text and create input_ids and attention_mask tensors
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=BLOCK_SIZE,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids'][0]

    return input_ids.unsqueeze(0)

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    model = EncoderTransformer(len(tokenizer.vocab), EMBEDDING_DIM, BLOCK_SIZE, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    app.run()
