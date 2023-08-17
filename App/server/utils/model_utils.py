import os
import torch
# from google.cloud import storage
from models.model_schema.model import EncoderTransformer
from transformers import AutoTokenizer
from .preprocessing_utils import preprocessing_fn


topics = [
    'Bank Account Services', 
    'Credit card / Prepaid card', 
    'Others', 
    'Theft / Dispute reporting', 
    'Mortgages / Loans'
]

def load_model(model_type, vocab_size, embedding_dim, block_size, num_classes):
    # bucket = 'ticket-classifier-models-bucket'
    filename = f'text_classification_{model_type}.pth'
    path = f'models/trained_models/' + filename

    # client = storage.Client()
    # bucket = client.bucket(bucket)
    # blob = bucket.blob(filename)
    # blob.download_to_filename(path)

    model = EncoderTransformer(vocab_size, embedding_dim, block_size, num_classes)
    model.load_state_dict(torch.load(path))

    return model

def load_tokenizer():
    return AutoTokenizer.from_pretrained('bert-base-uncased')

def classify_text(text, model, model_type, tokenizer, block_size):
    input_ids = preprocess_and_encode_text(text, tokenizer, block_size, model_type=model_type)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        predicted_class = torch.argmax(outputs, dim=1).item()
    return topics[predicted_class], predicted_class

def preprocess_and_encode_text(text, tokenizer, block_size, model_type='no-pos'):
    text = preprocessing_fn(text, model_type=model_type)
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=block_size,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )
    return encoding['input_ids'][0].unsqueeze(0)
