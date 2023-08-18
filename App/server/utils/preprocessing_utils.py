import regex as re
import en_core_web_sm

def preprocessing_fn(x, model_type=None):
    """
    Preprocess input text based on the specified model type. 
    Text cleaning is model agnostic, adds lemmatization and POS extraction if necessary.

    Args:
        x (str): Input text to preprocess.
        model_type (str, optional): Model type. Defaults to None.

    Returns:
        str: Processed text.
    """
    x = clean(x)
    
    if model_type == 'pos':
        x = lemmatize(x)
        x = noun_extraction(x)

    return x


def clean(text):
    """
    Clean and preprocess text by converting to lowercase, removing non-alphanumeric characters,
    digits, and common placeholders.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    import re

    text = text.lower()
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.replace('xxxx', '')
    text = text.strip()
    return text

def lemmatize(text):
    """
    Lemmatize text using spaCy.

    Args:
        text (str): Input text.

    Returns:
        str: Lemmatized text.
    """
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    return ' '.join(tok.lemma_ for tok in doc)

def noun_extraction(text):
    """
    Extract nouns from text using TextBlob.

    Args:
        text (str): Input text.

    Returns:
        str: Extracted nouns.
    """
    from textblob import TextBlob
    blob = TextBlob(text)
    return ' '.join(tok for (tok, tag) in blob.tags if tag == 'NN')
