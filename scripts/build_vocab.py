import sys
import os
os.environ["HF_HOME"] = "G:/huggingface_cache"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from datasets import load_dataset
from utils.preprocessing import clean_text, tokenize_text, build_vocab

def main():
    print("🔄 Loading WikiText-103 training split...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

    print("Cleaning and tokenizing...")
    token_lists = [
        tokenize_text(clean_text(example['text']))
        for example in dataset if example['text'].strip() != ''
    ]

    print("Building vocabulary...")
    word2idx, idx2word = build_vocab(token_lists, max_vocab_size=50000)

    print("Saving to data/vocab.json...")
    with open("data/vocab.json", "w") as f:
        json.dump(word2idx, f, indent=2)

    print("Done! Vocab size:", len(word2idx))

if __name__ == "__main__":
    main()