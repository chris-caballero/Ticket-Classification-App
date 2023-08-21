import sys
sys.path.append('../server')

from transformers import BertTokenizerFast
from models.model_schema.model import EncoderTransformer
from utils.model_utils import *
from utils.globals import *

# make sure we can load the tokenizer and model as expected
# this could use Mocks too if the application gets a little more complex
def test_load_tokenizer_and_model():
    tokenizer = load_tokenizer()
    model = load_model('no-pos', len(tokenizer.vocab), EMBEDDING_DIM, BLOCK_SIZE, NUM_CLASSES)
    assert isinstance(tokenizer, BertTokenizerFast)
    assert isinstance(model, EncoderTransformer)

