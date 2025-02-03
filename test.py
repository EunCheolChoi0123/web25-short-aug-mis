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

    # Prepare
    model_dir_list = sorted(os.listdir('./models'))

    # We'll store classification reports here
    classification_reports_df = pd.DataFrame(
        columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    )

    # Process each model
    for model_name in tqdm(model_dir_list, desc="Processing models", unit="model"):
        try:

            # Load cross-encoder, predict
            model = CrossEncoder(f'./models/{model_name}')

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

            # Construct rows for DataFrame
            report_rows = []
            for label, metrics in report_dict.items():
                # 'accuracy' is stored differently, so we handle it separately
                if label == 'accuracy':
                    report_rows.append({
                        'Size': model_name.split('_')[2],
                        'Augment': model_name.split('_')[1],
                        'Class': label,
                        'Precision': None,
                        'Recall': None,
                        'F1-Score': metrics,  # store accuracy in the F1-Score field
                        'Support': None
                    })
                # 'macro avg' or 'weighted avg'
                elif label in ['macro avg', 'weighted avg']:
                    report_rows.append({
                        'Size': model_name.split('_')[2],
                        'Augment': model_name.split('_')[1],
                        'Class': label,
                        **metrics
                    })
                else:
                    # Normal class name (oppose, support, neither)
                    report_rows.append({
                        'Size': model_name.split('_')[2],
                        'Augment': model_name.split('_')[1],
                        'Class': label,
                        **metrics
                    })

            temp_df = pd.DataFrame(report_rows)
            # Final post-processing
            # Move 'accuracy' from the 'F1-Score' column to an 'accuracy' column
            temp_df['accuracy'] = temp_df['F1-Score']
            temp_df.drop(
                columns=['Precision', 'Recall', 'F1-Score', 'Support'],
                errors='ignore',
                inplace=True
            )

            # Append report to the main DataFrame
            classification_reports_df = pd.concat([classification_reports_df, temp_df])



        except Exception as e:
            print(e, flush=True)

    classification_reports_df = classification_reports_df.drop(['Precision', 'Recall', 'F1-Score', 'Support'], axis=1)
    classification_reports_df['f1-score'] = classification_reports_df.apply(lambda row: row['accuracy'] if pd.notna(row['accuracy']) else row['f1-score'], axis=1)
    classification_reports_df.drop(columns=['accuracy'], inplace=True)
    classification_reports_df.to_csv(
        f'./eval/classification_reports.csv',
        index=False
    )
    
    print("All classification reports saved successfully.", flush=True)

if __name__ == "__main__":

    # Create a single directory
    try:
        os.mkdir("./eval")
        print("Directory created successfully.")
    except FileExistsError:
        print("Directory already exists.")
        
    main()
