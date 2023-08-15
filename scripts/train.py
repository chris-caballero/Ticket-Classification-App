import warnings
import os
import torch
import logging
import multiprocessing
import pandas as pd
from transformers import AutoTokenizer
from utils.data import TicketDataset, to_dataloader
from utils.models import EncoderTransformer
from utils.model_utils import train_and_evaluate

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

save_models = True
TICKETS_PATH = '../data/preprocessed_labeled_complaints.pkl'
TRAINED_MODEL_PATH = '../trained_models/text_classification_model.pth'

# HYPER-PARAMETERS
epochs = 5
n_embed = 32
block_size = 200
num_classes = 5
num_filters = 512
embedding_dim = 300
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logging.info(f'Device: {device}')
logging.info('Loading Tickets Dataset')

# Loads the processed an d labeled support tickets from file
complaints = pd.read_pickle(TICKETS_PATH)

# Load the tokenizer and create the dataset using our TicketDataset class
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = TicketDataset(complaints, tokenizer)
trainset, testset = to_dataloader(dataset, split=0.8)
vocabulary_size = len(dataset.tokenizer.vocab)

logging.info('Done!')

# Get the appropriate class weights for the cnn model
labels = torch.tensor([val for val in complaints['label']])

if __name__ == '__main__':
    multiprocessing.freeze_support()

    logging.info('Creating Models')

    # MODEL INITIALIZATION
    model = EncoderTransformer(vocabulary_size, embedding_dim, block_size, num_classes).to(device)
    
    logging.info('Done!')
    logging.info('Training and Evaluating Models')

    # TRAINING AND EVALUATION BLOCK
    train_and_evaluate(model, trainset, testset, epochs=5, verbose=True, device=device)

    logging.info('Done!')


    # SAVE MODELS
    if save_models:
        logging.info('Saving Models')
        torch.save(model.state_dict(), TRAINED_MODEL_PATH)
        logging.info('Done!')


