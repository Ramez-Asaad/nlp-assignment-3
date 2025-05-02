from datasets import load_dataset
from preprocessing import encode_sentence
import numpy as np
from transformers import AutoTokenizer

def load_wikitext(split='train', max_length=50, vocab_size=50000):
    # Load dataset from Hugging Face
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)

    # Use a pretrained tokenizer for fast tokenization & vocab handling
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or use "bert-base-uncased"

    # Limit vocab size
    tokenizer.model_max_length = max_length
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type='torch', columns=['input_ids'])
    return tokenized_dataset, tokenizer
  
def load_dataset_from_txt(path, word2idx, max_len=50):
    sequences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            seq = encode_sentence(tokens, word2idx, max_len)
            sequences.append(seq)
    return np.array(sequences)

