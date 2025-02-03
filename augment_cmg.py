import pandas as pd

import openai
from openai import OpenAI

import json
import random
from tqdm import tqdm
tqdm.pandas()

import argparse

import datetime

from transformers import pipeline
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login

client = OpenAI(
  api_key='YOUR_OPENAI_TOKEN',
)

login(token="YOUR_HUGGINGFACE_TOKEN")

def create_prompt(df, row, size, system_prompt, task_description, shot=3):
    """
    Creates a prompt for the OpenAI API based on the input data.
    """

    label = row['label']
    pair_id = row['id']
    
    # Randomly select 3 examples from the same label and within the size limit
    examples = df[(df['label'] == label) & (df['train_size'] <= size) & (df['id'] != id)].sample(n=shot).to_dict('records')
    system = task_description[label] + "\n" + system_prompt

    input = ''

    for example in examples:
        input += "CLAIM: " + example['claim'] + "\n"
        input += "TWEET: " + example['tweet'] + "\n\n"

    input += "CLAIM: " + row['claim'] + "\n"
    input += "TWEET: "

    return {"system": system, "input": input, "id": pair_id}

def generate_output(prompt, model, temperature=.5, max_tokens=512):

    system = prompt['system']
    input = prompt['input']

    if model[:3] == 'gpt':

        # Generate the response using OpenAI's API
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": input},
            ]
        )
        return response.choices[0].message.content.strip()

    else:

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
        ]

        outputs = generation(
            messages,
            max_new_tokens=256,
            temperature=0.5,
            do_sample=True,
            pad_token_id = generation.tokenizer.eos_token_id
        )
        generated_output = outputs[0]['generated_text'][-1]['content']

        if "Tweet: " in generated_output:  # Correct syntax for substring check
            return generated_output.split('Tweet: ')[1].strip()
        elif "TWEET: " in generated_output:  # Correct syntax for substring check
            return generated_output.split('TWEET: ')[1].strip()
        else:
            return generated_output.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run augmentation with a specific model")
    parser.add_argument("-m", "--model", type=str, required=True, help="Specify the model to use (e.g., qwen, llama, gpt-4o, gpt-3.5-turbo)")
    
    args = parser.parse_args()
    model_name = args.model

    df = pd.read_json("./dataset.json")
    df['id'] = df.index

    system_prompt = "Try your best to mimic the styles of example TWEET.\nRespond with only a single TWEET."

    task_description = {'support': """Generate TWEET so that if TWEET is true, then CLAIM is also true.""",
                        'oppose': """Generate TWEET so that if TWEET is true, then CLAIM is false.""",
                        'neither': """Generate TWEET so that if TWEET is true, then CLAIM cannot be said to be neither true nor false."""}

    prompt_list_1 = []
    prompt_list_2 = []
    prompt_list_3 = []
    prompt_list_4 = []

    for index, row in tqdm(df.iterrows()):
        size = row['train_size']

        prompt_1 = create_prompt(df, row, size, system_prompt, task_description)
        prompt_list_1.append(prompt_1)

        prompt_2 = create_prompt(df, row, size, system_prompt, task_description)
        prompt_list_2.append(prompt_2)

        prompt_3 = create_prompt(df, row, size, system_prompt, task_description)
        prompt_list_3.append(prompt_3)

        prompt_4 = create_prompt(df, row, size, system_prompt, task_description)
        prompt_list_4.append(prompt_4)

    if model_name[:3] != 'gpt':

        if model_name=="llama":
            model_id = "meta-llama/Llama-3.1-8B-Instruct"
        elif model_name=="qwen":
            model_id="Qwen/Qwen2.5-7B-Instruct"
        
        generation = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="cuda",
        )

    if model_name == "gpt-4o":
        model_col = "gpt4"
    elif model_name == "gpt-3.5-turbo":
        model_col = "gpt3"
    else:
        model_col = model_name

    for i, row in tqdm(df.iterrows()):

        df.loc[i, f'{model_col}_1'] = generate_output(prompt_list_1[i], model=model_name)
        df.loc[i, f'{model_col}_2'] = generate_output(prompt_list_2[i], model=model_name)
        df.loc[i, f'{model_col}_3'] = generate_output(prompt_list_3[i], model=model_name)
        df.loc[i, f'{model_col}_4'] = generate_output(prompt_list_4[i], model=model_name)