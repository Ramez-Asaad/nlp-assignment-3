import re
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from typing import List, Tuple
import json

def clean_text(text: str) -> str:
    """
    Cleans the input text by lowercasing the text, removing unwanted characters and normalizing whitespace.
    """
    text= text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes the input text into words using NLTK's word_tokenize.
    """
    return word_tokenize(text)



def build_vocab(token_lists: List[List[str]], max_vocab_size: int = 50000) -> Tuple[dict, dict]:
    """
    Builds a word-to-index and index-to-word vocab dictionary.
    """
    all_tokens = [token for tokens in token_lists for token in tokens]
    most_common = Counter(all_tokens).most_common(max_vocab_size)

    word2idx = {word: idx+2 for idx, (word, _) in enumerate(most_common)}  # +2 for <PAD> and <UNK>
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word



def encode(tokens: List[str], word2idx: dict) -> List[int]:
    """
    Converts list of tokens to list of indices.
    """
    return [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]



def generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str]]:
    """
    Generates n-grams from a list of tokens.
    """
    return list(ngrams(tokens, n))
  
  
  
def pad_sequence(seq: List[int], max_length: int, pad_idx: int = 0) -> List[int]:
    """
    Pads or truncates a sequence to fixed length.
    """
    if len(seq) < max_length:
        return seq + [pad_idx] * (max_length - len(seq))
    return seq[:max_length]

''''
def get_word2idx():
    """
    Loads word2idx mapping from a JSON file.
    """
    with open("G:\\OneDrive - Alamein International University\\Uni stuff\\semester 6 - Spring 24-25\\NLP\\assignments\\assignment 3 v0.1\\nlp-assignment-3\\data\\vocab.json", "r") as f:
        word2idx = json.load(f)
    return word2idx
  
  

word2idx = get_word2idx()
vocab_size = len(word2idx)
pad_token = word2idx["<PAD>"]
unk_token = word2idx["<UNK>"]


def encode_sentence(tokens, word2idx, max_len=50):
    ids = [word2idx.get(token, unk_token) for token in tokens]
    ids = ids[:max_len]
    ids += [pad_token] * (max_len - len(ids))
    return ids
  '''