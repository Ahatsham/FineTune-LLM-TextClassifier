# Training logic will go here
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from model.utils import load_config
from data.preprocess import load_and_preprocess

cfg = load_config("configs/config.yaml")
model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=cfg["num_labels"])
dataset = load_and_preprocess(cfg["model_name"])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=cfg["num_epochs"],
    per_device_train_batch_size=cfg["train_batch_size"],
    per_device_eval_batch_size=cfg["eval_batch_size"],
    load_best_model_at_end=True,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()