import torch
# from google.cloud import storage
from models.model_schema.model import EncoderTransformer
from transformers import AutoTokenizer
from .preprocessing_utils import preprocessing_fn
import os
from pathlib import Path

# Define a module-level variable for the models directory
MODELS_DIR = "models/trained_models"

# Construct an absolute path to the models directory
project_root = Path(__file__).resolve().parent.parent  # Adjust this based on your project structure
models_path = project_root / MODELS_DIR

topics = [
    'Bank Account Services',
    'Credit card / Prepaid card',
    'Others',
    'Theft / Dispute reporting',
    'Mortgages / Loans'
]

def load_model(model_type, vocab_size, embedding_dim, block_size, num_classes):
    """
    Load a trained model.

    Args:
        model_type (str): Type of the model to load.
        vocab_size (int): Vocabulary size for the model.
        embedding_dim (int): Embedding dimension for the model.
        block_size (int): Block size for the model.
        num_classes (int): Number of classes for classification.

    Returns:
        torch.nn.Module: Loaded model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = f'text_classification_{model_type}.pth'
    path = models_path / filename

    model = EncoderTransformer(vocab_size, embedding_dim, block_size, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))

    return model

def load_tokenizer():
    """
    Load a pre-trained tokenizer.

    Returns:
        transformers.tokenization_utils.PreTrainedTokenizer: Loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained('bert-base-uncased')

def classify_text(text, model, model_type, tokenizer, block_size):
    """
    Classify input text using the provided model.

    Args:
        text (str): Input text to classify.
        model (torch.nn.Module): Loaded classification model.
        model_type (str): Type of the model used.
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer): Loaded tokenizer.
        block_size (int): Maximum block size for tokenization.

    Returns:
        tuple: Tuple containing the predicted topic and its class index.
    """
    input_ids = preprocess_and_encode_text(text, tokenizer, block_size, model_type=model_type)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        predicted_class = torch.argmax(outputs, dim=1).item()
    return topics[predicted_class], predicted_class

def preprocess_and_encode_text(text, tokenizer, block_size, model_type='no-pos'):
    """
    Preprocess and encode input text using a tokenizer.

    Args:
        text (str): Input text to preprocess.
        tokenizer (transformers.tokenization_utils.PreTrainedTokenizer): Loaded tokenizer.
        block_size (int): Maximum block size for tokenization.
        model_type (str, optional): Type of the model used. Defaults to 'no-pos'.

    Returns:
        torch.Tensor: Encoded input tensor.
    """
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
