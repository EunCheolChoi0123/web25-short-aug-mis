import os
import csv
import gzip
import logging
import math
from datetime import datetime
import torch

from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEF1Evaluator
from sentence_transformers.readers import InputExample

# Function to load data based on dataset inclusion
def load_data(nli_dataset_path):
    train_samples = []
    dev_samples = []
    with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            label_id = label2int[row["label"]]
            if row["split"] == "train":
                train_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label_id))
            else:
                dev_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label_id))
    return train_samples, dev_samples

if __name__ == "__main__":
    
    # Define datasets and labels
    nli_dataset_path = "./data/nli.tsv.gz"
    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    
    # Define training parameters
    train_batch_size = 16  # Define batch size
    num_epochs = 5  # Define number of epochs
    
    # Load data for the current NLI type
    train_samples, dev_samples = load_data(nli_dataset_path)
    
    # Define model save path
    model_save_path = "./pretrained_ce"
    
    # Create a single directory
    try:
        os.mkdir(model_save_path)
        print("Directory created successfully.")
    except FileExistsError:
        print("Directory already exists.")
    
    # Define CrossEncoder model
    model = CrossEncoder(f"microsoft/deberta-v3-base", num_labels=len(label2int), max_length=256)
    
    # Create DataLoader for training samples
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    
    # Define evaluators
    evaluator = CEF1Evaluator.from_input_examples(dev_samples, name="nli-dev")
    
    # Calculate warmup steps
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    logger.info(f"Warmup-steps: {warmup_steps}")
    
    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=30000,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
    )