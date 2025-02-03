from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import torch

if __name__ == "__main__":

    # load dataset and model
    df=pd.read_json('./data/dataset.json')
    tokenizer = AutoTokenizer.from_pretrained("protectai/distilroberta-base-rejection-v1")
    model = AutoModelForSequenceClassification.from_pretrained("protectai/distilroberta-base-rejection-v1")

    # select columns that contain cmg
    settings = [
        'gpt4_1', 'gpt4_2', 'gpt4_3', 'gpt4_4',
        'gpt3_1', 'gpt3_2', 'gpt3_3', 'gpt3_4',
        'llama_1', 'llama_2', 'llama_3', 'llama_4',
            'qwen_1', 'qwen_2', 'qwen_3', 'qwen_4',
    ]

    for column_name in settings:
        batch_size = 128 # adjust based on your gpu resource

        for i in tqdm(range(0, len(df[column_name]), batch_size)):
            batch_texts = df[column_name][i:i+batch_size].tolist()
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            rejection_probs = probs[:, 1].tolist()  # Extract probability for class 1 (rejection)

            df.loc[i:i+batch_size-1, f'rj_{column_name}'] = rejection_probs

    df.to_json('./data/dataset.json', orient='records', indent=4)