import warnings
import os
import pandas as pd
import torch
import logging
import multiprocessing
from transformers import AutoTokenizer
from utils.data import TicketDataset, to_dataloader
from utils.models import BertClassifier, ConvNet, EncoderTransformer
from utils.model_utils import get_class_weights, train, evaluate, train_and_evaluate, cnn_train_eval, transformer_train_eval

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

save_models = False
support_tickets_file = '../data/preprocessed_labeled_complaints.pkl'

# HYPER-PARAMETERS
epochs = 5
block_size = 200
n_embed = 32
num_ticket_classes = 5
num_filters = 512
embedding_dim = 300
filter_sizes = [5, 4, 3]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logging.info(f'Device: {device}')
logging.info('Loading Tickets Dataset')

# Loads the processed an d labeled support tickets from file
complaints = pd.read_pickle(support_tickets_file)

# Load the tokenizer and create the dataset using our TicketDataset class
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = TicketDataset(complaints, tokenizer)
trainset, testset = to_dataloader(dataset, split=0.8)
vocabulary_size = len(dataset.tokenizer.vocab)

logging.info('Done!')

# Get the appropriate class weights for the cnn model
labels = torch.tensor([val for val in complaints['label']])
class_weights = get_class_weights(labels, num_ticket_classes, mode=2)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    logging.info('Creating Models')

    # MODEL INITIALIZATION
    
    # HAN = HierarchicalAttentionNetwork(vocabulary_size, embedding_dim, num_filters)
    # transformer = EncoderTransformer(vocabulary_size, embedding_dim, block_size, num_ticket_classes).to(device)
    bert = BertClassifier(num_classes=num_ticket_classes).to(device)
    # cnn = ConvNet(vocabulary_size, embedding_dim, num_filters, filter_sizes, num_ticket_classes).to(device)
    
    logging.info('Done!')
    logging.info('Training and Evaluating Models')

    # # TRAINING AND EVALUATION BLOCK
    train(bert, trainset, is_transformer=False, class_weights=None, device=device, epochs=5)
    acc = evaluate(bert, testset, is_transformer=False, device=device)
    print(f'Validation Accuracy:', acc)
    # cnn = train_and_evaluate(cnn_train_eval, cnn, trainset, testset, class_weights=class_weights, epochs=1, device=device)
    # transformer = train_and_evaluate(transformer_train_eval, transformer, trainset, testset, is_transfomer=True, epochs=1, device=device)

    logging.info('Done!')

    # SAVE MODELS
    if save_models:
        logging.info('Saving Models')
        # torch.save(cnn.state_dict(), '../models/cnn-ticket-classifier.pt')
        torch.save(HAN.state_dict(), '../models/transformer-ticket-classifier.pt')
        logging.info('Done!')


