import pandas as pd
import numpy as np
import random

import re
import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
import networkx as nx

from tqdm import tqdm

def clean_string(text: str) -> str:

    # Remove special characters at the beginning and end
    text = re.sub(r'@\S+', '@username', text)
    text = re.sub(r'@username', '', text)  # End
    text = re.sub(r'^[^a-zA-Z0-9]+', '', text)  # Beginning
    text = re.sub(r'[^a-zA-Z0-9]+$', '', text)  # End

    # Remove URLs
    text = re.sub(r'https?://\S+\b', '', text)
    text = re.sub(r'http://\S+\b', '', text)

    # Replace '[NEWLINE]' with actual newline character
    text = text.replace('[NEWLINE]', '\n')

    return text

def lemmatize_sentence(sentence: str) -> str:

    lemmatizer = WordNetLemmatizer()
    sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
    if not isinstance(sentence, str):
        return ""
    tokens = sentence.lower().split()  # Lowercase and split into tokens
    lem_tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lemmatize each token
    return " ".join(lem_tokens)  # Rejoin lemmatized tokens into a sentence

def find_overlap_rows(input_df, col1, col2):

    """
    Find rows where the lemmatized content of col1 and col2 have 100% overlap.
    """

    df = input_df.copy()
    rows_with_full_overlap = []

    # Lemmatize the columns
    df[col1] = df[col1].apply(lemmatize_sentence)
    df[col2] = df[col2].apply(lemmatize_sentence)

    # Row-by-row comparison
    for index, row in df.iterrows():
        words_col1 = set(row[col1].split())
        words_col2 = set(row[col2].split())
        
        # Find minimum counts of common words and total words
        common_words = words_col1.intersection(words_col2)
        all_words = words_col1.union(words_col2)

        # Check if overlap is 100%
        if len(all_words) > 0:
            overlap = len(common_words) / len(all_words)
            if overlap == 1.:
                rows_with_full_overlap.append(index)

    # Return the filtered DataFrame
    return rows_with_full_overlap

def preprocess_and_build_graph(
    df: pd.DataFrame,
    overlap_threshold: float = 0.9,
    random_seed: int = 42
):
    """
    (1) Concatenate 'claim' and 'tweet' columns for overlap comparison.
    (2) Lemmatize the concatenated text.
    (3) Compute word-overlap between rows based on the concatenated text.
    (4) Build and return a graph G where edges connect rows exceeding the overlap_threshold.
    """

    np.random.seed(random_seed)
    random.seed(random_seed)

    # Step 1: Create concatenated column
    df_out = df.copy()
    df_out['concatenated_text'] = (df_out['claim'].astype(str) + ' ' + df_out['tweet'].astype(str)).apply(lemmatize_sentence)

    # Step 2: Convert lemmatized text to sets of words
    word_sets = df_out['concatenated_text'].apply(lambda x: set(x.split())).tolist()

    def compute_overlap(i, j):
        """
        Returns the overlap between word sets of row i and j.
        Overlap = intersection_of_words / min(len(row_i_words), len(row_j_words))
        """
        set_i = word_sets[i]
        set_j = word_sets[j]
        intersection_size = len(set_i.intersection(set_j))
        union_size = len(set_i.union(set_j))

        if union_size == 0:
            return 0.0
        return intersection_size / union_size

    # Step 3: Build graph where nodes are row indices, and edges exist if overlap > threshold
    G = nx.Graph()
    G.add_nodes_from(df_out.index)

    indices = df_out.index.tolist()
    n = len(indices)

    # O(N^2) approach: for each pair, compute overlap and add edge if > threshold
    for idx_a in tqdm(range(n)):
        i = indices[idx_a]
        for idx_b in range(idx_a + 1, n):
            j = indices[idx_b]
            overlap_val = compute_overlap(idx_a, idx_b)
            if overlap_val > overlap_threshold:
                G.add_edge(i, j)

    return G

def inspect_split(df, G):

    # List to store edges where 'split' values differ
    overlapping_rows_between_splits = []

    # Iterate through all edges in G
    for edge in G.edges():
        node1, node2 = edge
        
        # Compare the 'split' column for both nodes
        if df.loc[node1, 'split'] != df.loc[node2, 'split']:
            overlapping_rows_between_splits.append([edge, (df.loc[node1, 'split'], df.loc[node2, 'split'])])

    return overlapping_rows_between_splits
    
if __name__ == "__main__":
    df = pd.read_json('./data/dataset.json')
    print('Overlapping rows:')
    print(find_overlap_rows(df, 'claim', 'tweet'))
    G = preprocess_and_build_graph(df)
    print(inspect_split(df, G))