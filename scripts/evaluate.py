import torch
import logging
import pandas as pd
import multiprocessing
from transformers import AutoTokenizer
from utils.data import TicketDataset, to_dataloader
from utils.models import EncoderTransformer
from utils.model_utils import evaluate

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='../model_performance_2.log'
)

# FILES
TICKETS_FILE = '../data/preprocessed_labeled_complaints.pkl'
MODEL_FILE = '../trained_models/text_classification_model.pth'


# HYPER-PARAMETERS
epochs = 5
n_embed = 32
block_size = 200
num_classes = 5
num_filters = 512
embedding_dim = 300
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logging.info('Loading Tickets Dataset')

# Loads the processed and labeled support tickets from file
complaints = pd.read_pickle(TICKETS_FILE)

# Sets up the DataLoaders
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = TicketDataset(complaints, tokenizer)
dataloader = to_dataloader(dataset, split=1)
vocabulary_size = len(dataset.tokenizer.vocab)

logging.info('Done!')

if __name__ == '__main__':
    multiprocessing.freeze_support()

    logging.info('Loading Models')

    model = EncoderTransformer(vocabulary_size, embedding_dim, block_size, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_FILE))

    logging.info('Done!')
    logging.info('Evaluating Models')
    
    accuracy = evaluate(model, dataloader, device=device)

    print('-'*50 + '\nTRANSFORMER\n' + '-'*50)
    print('Accuracy:', accuracy)


