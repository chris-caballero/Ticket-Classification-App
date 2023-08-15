import torch
import logging
import pandas as pd
import multiprocessing
from transformers import AutoTokenizer
from utils.data import TicketDataset
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
cut = 0
epochs = 5
num_ticket_classes = 5
num_filters = 512
embedding_dim = 300
block_size = 200
filter_sizes = [5, 4, 3]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logging.info('Loading Tickets Dataset')
# Loads the processed and labeled support tickets from file
complaints = pd.read_pickle(TICKETS_FILE)
complaints = complaints[:-int(cut*len(complaints))] if cut else complaints

# Sets up the DataLoaders
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = TicketDataset(complaints, tokenizer, field='complaint', block_size=block_size)
vocabulary_size = len(dataset.tokenizer.vocab)

logging.info('Done!')


logging.info('Loading Models')

model = EncoderTransformer(vocabulary_size, embedding_dim, block_size, num_ticket_classes).to(device)
model.load_state_dict(torch.load(MODEL_FILE))

logging.info('Done!')

labels = torch.tensor([val for val in complaints['label']])
if __name__ == '__main__':
    multiprocessing.freeze_support()

    logging.info('Evaluating Models')
    
    test_accuracy = evaluate(model, dataset, device=device)

    print('-'*50 + '\nTRANSFORMER\n' + '-'*50)
    print('Validation Accuracy:', test_accuracy)


