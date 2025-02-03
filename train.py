import pandas as pd
import os
import math
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sklearn.metrics import classification_report
import torch.nn as nn
import torch

def run_experiment(model_huggingface_path, task, total_sample_size_list, augment, max_seq_length=256, repeat=1):
    
    run_count = 0
    while run_count < repeat:
        augment_column_list = []
        if augment==True:
            if task == 'bt':
                augment_column_list = ['bt_de', 'bt_ru', 'bt_zh', 'bt_it']
            else:
                augment_column_list = [f'{task}_1', f'{task}_2', f'{task}_3', f'{task}_4'] 
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        freq = df[df['split']=='train']['label'].value_counts(normalize=True)
        for train_size in total_sample_size_list:
            task_name = f"{task}_{train_size}"
            print(task_name)
    
            
    
            label2int = {"oppose": 0, "support": 1, "neither": 2}
            int2label = {v: k for k, v in label2int.items()}
            train_batch_size = 32
            num_epochs = 5
    
            # Define class weights
            class_weights = torch.tensor([float(1/freq["oppose"]), float(1/freq["support"]), float(1/freq["neither"])])
            class_weights = class_weights.to(device)
    
            def get_samples(df):
                train_samples = []
                dev_samples = df[df['split'] == 'dev']
    
                # Finetuning sample size
                train_df = df[(df['split'] == 'train') & (df['train_size'].astype(int) <= train_size)]
    
                # Check the distribution of labels in the sampled DataFrame
                print(train_df['label'].value_counts(normalize=True))
    
                for index, row in train_df.iterrows():
                    train_samples.append(InputExample(texts=[row['tweet'], row['claim']], label=label2int[row['label'].lower()]))
                    if augment==True:
                        for augment_column in augment_column_list:
                            if ("gpt" in augment_column or "llama" in augment_column or "qwen" in augment_column) and row[f'rj_{augment_column}'] < .9:
                                train_samples.append(InputExample(texts=[row[augment_column], row['claim']], label=label2int[row['label'].lower()]))
                            else:
                                train_samples.append(InputExample(texts=[row[augment_column], row['claim']], label=label2int[row['label'].lower()]))
    
                dev_samples = [InputExample(texts=[row['tweet'], row['claim']], label=label2int[row['label'].lower()]) for _, row in dev_samples.iterrows() if pd.notna(row['tweet']) and pd.notna(row['claim'])]
    
                return train_samples, dev_samples
    
            def evaluate_model(model, dev_samples, epoch):
                texts = [(sample.texts[0], sample.texts[1]) for sample in dev_samples]
                true_labels = [sample.label for sample in dev_samples]
    
                # Get predictions from the model
                pred_scores = model.predict(texts)
                pred_labels = pred_scores.argmax(axis=1)
    
                # Generate classification report
                report = classification_report(true_labels, pred_labels, target_names=[int2label[i] for i in range(len(label2int))], digits=4)
                print(f"Epoch {epoch + 1} Classification Report:\n{report}")
    
            def train_model():
                print(f"Training {task_name}...")
    
                # Get samples based on split column
                train_samples, dev_samples = get_samples(df)
    
                # Initialize model without specifying loss function
                model = CrossEncoder(model_huggingface_path, num_labels=len(label2int), max_length=max_seq_length)
    
                # Create DataLoader
                train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    
                # Calculate warmup steps
                warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
                print(f"Warmup-steps: {warmup_steps}")
    
                # Define model save path
                model_save_path = f"./models/finetuned_{task_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
                # Custom loss function with class weights
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    
                # Train the model using custom loss function within fit method
                for epoch in range(num_epochs):
                    model.fit(
                        train_dataloader=train_dataloader,
                        epochs=1,
                        warmup_steps=warmup_steps,
                        output_path=model_save_path,
                        loss_fct=loss_fct  # Specify custom loss function here
                    )
    
                # Save the model after each epoch
                model.save(model_save_path)
                print(f"Model saved to {model_save_path}")
    
                # Evaluate the model after each epoch
                evaluate_model(model, dev_samples, epoch)
    
            # Train with 'split'
            train_model()
        
        run_count +=1

if __name__ == "__main__":
  
    df = pd.read_json("./data/dataset.json")

    # Create a single directory
    try:
        os.mkdir("./models")
        print("Directory created successfully.")
    except FileExistsError:
        print("Directory already exists.")

    total_sample_size_list = [100, 200, 500, 1000, 2000, 5000]
    model_huggingface_path = './pretrained_ce'

    for task in ["base", "bt", "aeda", "eda", "gpt4", "gpt3", "llama", "qwen"]:

        augment = True
        if task == "base":
            augment = False
        
        run_experiment(model_huggingface_path, task, total_sample_size_list, augment, 5)
