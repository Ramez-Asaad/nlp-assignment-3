
import sys
import os
###path of models folder
sys.path.append(os.path.abspath('G:\\OneDrive - Alamein International University\\Uni stuff\\semester 6 - Spring 24-25\\NLP\\assignments\\assignment 3 v0.1\\nlp-assignment-3\\models'))
###path of utils folder
sys.path.append(os.path.abspath('G:\\OneDrive - Alamein International University\\Uni stuff\\semester 6 - Spring 24-25\\NLP\\assignments\\assignment 3 v0.1\\nlp-assignment-3\\utils'))
from datasets import load_dataset
from preprocessing import clean_text, tokenize_text

def save_split(split_name):
    print(f"🔄 Processing split: {split_name}")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split_name)
    output_lines = []

    for example in dataset:
        text = example['text'].strip()
        if text:
            cleaned = clean_text(text)
            tokens = tokenize_text(cleaned)
            output_lines.append(" ".join(tokens))

    os.makedirs("data", exist_ok=True)
    with open(f"data/{split_name}.txt", "w") as f:
        f.write("\n".join(output_lines))

    print(f"✅ Saved to data/{split_name}.txt")

def main():
    for split in ["train", "validation", "test"]:
        save_split(split)

if __name__ == "main":
    main()
