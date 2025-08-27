import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

def load_summarization_model(model_path="./models/bart_summarizer", device=-1):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
    return summarizer

def load_classification_model(model_path="./models/distilbert_agnews", device=-1):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    return classifier

def run_infer(task, text, device):
    if task == "summarization":
        summarizer = load_summarization_model(device=device)
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    elif task == "classification":
        classifier = load_classification_model(device=device)
        prediction = classifier(text)
        return prediction
    else:
        raise ValueError("Task must be either 'summarization' or 'classification'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["summarization", "classification"])
    parser.add_argument("text", type=str)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Select device: cpu or gpu")
    args = parser.parse_args()

    # map device
    device = 0 if (args.device == "gpu" and torch.cuda.is_available()) else -1

    output = run_infer(args.task, args.text, device)
    print(output)
