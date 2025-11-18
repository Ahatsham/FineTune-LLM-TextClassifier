# 🧠 FineTune-LLM-Text Classifier

A clean and modular pipeline to fine-tune **pretrained Large Language Models (LLMs)** for **text classification** tasks.  
This repo uses the AG News dataset and **DistilBERT** for demonstration.

---

## 🚀 Features

- 🔥 Fine-tune any HuggingFace Transformer model
- 📚 Uses `AG News` dataset (4 classes: World, Sports, Business, Sci/Tech)
- 📈 Evaluation with accuracy and F1 score
- 🧪 Easy inference on new samples
- 🔧 Configurable hyperparameters
- 🛠️ Ready to extend with new datasets/models

---

## 🧩 Tech Stack

- 🤗 `transformers`
- 🤗 `datasets`
- 🔬 `scikit-learn`
- ⚙️ `PyTorch` backend (via Trainer API)

---

## 📦 Setup

```bash
git clone https://github.com/yourusername/FineTune-LLM-TextClassifier.git
cd FineTune-LLM-TextClassifier
pip install -r requirements.txt
