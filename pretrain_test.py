import os
import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def main():

    # Load data
    df = pd.read_json("./data/dataset.json")
    test_df = df[
        (df['split'] == 'test')
        & (df['tweet'].notna())
        & (df['claim'].notna())
    ]

    # Load cross-encoder, predict
    model = CrossEncoder(f'./pretrained_ce')

    # Create list of (premise, hypothesis) or (tweet, claim)
    sentence_pairs = list(zip(test_df['tweet'], test_df['claim']))

    # Predict: CrossEncoder returns class probabilities
    predictions = model.predict(sentence_pairs)

    # True labels
    true_labels = test_df['label'].tolist()

    # Argmax across columns
    predicted_numeric = np.argmax(predictions, axis=1)

    # Convert numeric labels -> string labels
    label_mapping = {1: 'support', 2: 'neither', 0: 'oppose'}
    predicted_labels = [label_mapping[num] for num in predicted_numeric]

    # Build classification report
    report_dict = classification_report(
        true_labels,
        predicted_labels,
        target_names=['neither', 'oppose', 'support'],
        output_dict=True
    )

    print(report_dict)

if __name__ == "__main__":
    main()
