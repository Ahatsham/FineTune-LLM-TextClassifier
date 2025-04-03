# Preprocessing logic will go here
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess(model_name: str, max_length: int = 256):
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset
