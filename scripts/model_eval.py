import torch
import logging
import pandas as pd
import multiprocessing
from transformers import AutoTokenizer
from utils.data import TicketDataset
from utils.models import TransformerClassifier, ConvNet, EncoderTransformer, HierarchicalAttentionNetwork
from utils.model_utils import CrossValidation
from utils.model_utils import get_class_weights, count_parameters, train

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='../model_performance_2.log'
)

# FILES
support_tickets_file = '../data/preprocessed_labeled_complaints.pkl'
cnn_file = '../models/cnn-ticket-classifier.pt'
bert_file = '../models/bert-ticket-classifier.pt'
transformer_file = '../models/transformer-ticket-classifier.pt'
quantized_file = '../models/quantized-transformer.pt'

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
complaints = pd.read_pickle(support_tickets_file)
complaints = complaints[:-int(cut*len(complaints))] if cut else complaints

# Sets up the DataLoaders
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = TicketDataset(complaints, tokenizer, field='complaint', block_size=block_size)
vocabulary_size = len(dataset.tokenizer.vocab)

logging.info('Done!')
logging.info('Loading Models')

# Load quantized model
# quantized_model = EncoderTransformer(vocabulary_size, embedding_dim, block_size, num_ticket_classes).to(device)
# quantized_model.load_state_dict(torch.load('path/to/quantized_model.pt'))

transformer = EncoderTransformer(vocabulary_size, embedding_dim, block_size, num_ticket_classes).to(device)
# cnn = ConvNet(vocabulary_size, embedding_dim, num_filters, filter_sizes, num_ticket_classes).to(device)
bert = TransformerClassifier(num_classes=num_ticket_classes).to(device)

HAN = HierarchicalAttentionNetwork(vocabulary_size, embedding_dim, num_filters)
# Loads models from saved state (trained)
# transformer.load_state_dict(torch.load(transformer_file))
# cnn.load_state_dict(torch.load(cnn_file))
# bert = load_state_dict(torch.load(bert_file))

logging.info('Done!')

labels = torch.tensor([val for val in complaints['label']])
class_weights = get_class_weights(labels, num_ticket_classes, mode=2)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    logging.info('Evaluating Models')

    # quantized_scores, quantized_crossval = CrossValidation(
    #     quantized_model, 
    #     dataset, 
    #     epochs=5, 
    #     verbose=False,
    #     device=device
    # )

    ### GPT HAN IMPLEMENTATION
    scores, crossval = CrossValidation(
        HAN, 
        dataset, 
        epochs=5, 
        verbose=True,
        device=device
    )
    logging.info(f'Scores: {scores}')
    logging.info(f'Cross Validation: {crossval}')

    # logging.info(f'Quantized Scores: {quantized_scores}')
    # logging.info(f'Quantized 10-Fold Cross Validation: {quantized_crossval}')


    # transformer_scores, transformer_crossval = CrossValidation(
    #     transformer, 
    #     dataset, 
    #     epochs=5, 
    #     verbose=True,
    #     device=device
    # )

    # logging.info(f'Transformer Scores: {transformer_scores}')
    # logging.info(f'Transformer 10-Fold Cross Validation: {transformer_crossval}')

    # print(f'Transformer Scores: {transformer_scores}')
    # print(f'Transformer 10-Fold Cross Validation: {transformer_crossval}')

    # # EVALUTATE TRANSFORMER CLASSIFIER
    # transformer_scores, transformer_crossval = CrossValidation(
    #     transformer, 
    #     dataset, 
    #     epochs=5, 
    #     verbose=False,
    #     device=device
    # )

    # logging.info(f'Transformer Scores: {transformer_scores}')
    # logging.info(f'Transformer 10-Fold Cross Validation: {transformer_crossval}')

    # print(f'Transformer Scores: {transformer_scores}')
    # print(f'Transformer 10-Fold Cross Validation: {transformer_crossval}')


    # # EVALUTATE CNN CLASSIFIER
    # cnn_scores, cnn_crossval = CrossValidation(
    #     cnn, 
    #     dataset, 
    #     epochs=5, 
    #     verbose=False,
    #     class_weights=class_weights, 
    #     device=device
    # )

    # logging.info(f'CNN Scores: {cnn_scores}')
    # logging.info(f'CNN 10-Fold Cross Validation: {cnn_crossval}')

    # print(f'CNN Scores: {cnn_scores}')
    # print(f'CNN 10-Fold Cross Validation: {cnn_crossval}')


    # EVALUATE BERT CLASSIFIER
    bert_scores, bert_crossval = CrossValidation(
        bert, 
        dataset, 
        epochs=5, 
        batch_size=16,
        verbose=True,
        is_transformer=True, 
        device=device
    )

    logging.info(f'Bert Scores: {bert_scores}')
    logging.info(f'Bert 10-Fold Cross Validation: {bert_crossval}')

    print(f'Bert Scores: {bert_scores}')
    print(f'Bert 10-Fold Cross Validation: {bert_crossval}')



