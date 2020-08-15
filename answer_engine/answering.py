import sys
sys.path.append('answer_engine/')
sys.path.append('../answer_engine/')

# ML dependencies
import torch
import transformers
import numpy as np
from transformers.data.metrics.squad_metrics import compute_predictions_logits

# Preprocessing
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import (
    SquadResult, 
    SquadV2Processor, 
    SquadExample)

# File management dependcies
import google_storage
import pickle
import json

import os
import re
import random
import time

# Imports for saving results
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='darkgrid')

# Initialize global paths
ANSWER_ENGINE_PATH = 'answer_engine/' if __name__ != '__main__' else '' # Changing the path to files in the answer_engine folder
MODEL_PATH = '' if __name__ != '__main__' else '../' # Changing the path to files in the model folder

# Answers a question based on context and a question with the specified model type
class question_answering:

    def __init__(self, model, tokenizer):
        # model: The preloaded Transformers model for question answering
        # tokenizer: The preloaded tokenizer that is used together with the transformers model

        self.model = model
        self.tokenizer = tokenizer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_batch_sz = 8

    # Runs the the prediction
    def __call__(self, contexts, question, max_len=384) -> tuple:
        # contexts: an iterable of the contexts possibly containing the answer to the question
        # question: The question we want answered
        # max_len: The maximum length of the contexts

        # Using the transformers own SQuAD preprocessing module
        examples = []
        for i, text in enumerate(contexts):
            example = SquadExample(
                qas_id=i,
                question_text=question, 
                context_text=text,
                answer_text=None,
                start_position_character=None,
                title='predict',
                answers=None,
                is_impossible=False)
            examples.append(example)

        # Creates a dataset which easily can be inputted into the model
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer, 
            max_seq_length=max_len,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            return_dataset='pt', 
            threads=len(examples),
            tqdm_enabled=False)

        # Sampels and converts the dataset to a dataloader
        batch_size = len(examples) if len(examples) < self.max_batch_sz else self.max_batch_sz
        
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        # Data variables
        results = []
        answer_scores = []

        # Sets the model into prediction mode
        self.model.eval()

        for batch in dataloader:
            # Turns the batch into device
            batch = [i.to(self.device) for i in batch]

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]}

                # The indices where the context is
                example_indices = batch[3]

                # Make a prediction
                outputs = self.model(**inputs)

                for i, examples_idx in enumerate(example_indices):
                    i_feature = features[examples_idx.item()]
                    unique_id = i_feature.unique_id

                    start_logits = outputs[0][i].detach().cpu().numpy()
                    end_logits = outputs[1][i].detach().cpu().numpy()

                    scores = score_answers(
                        start_logits, 
                        end_logits
                    ).scores(seqlen=max_len, n_best=1)
                    answer_scores.append(scores)

                    result = SquadResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits)
                    results.append(result)

        # Convert the results into actual predictions
        predictions = generate_answer_sent(
            results=results, 
            features=features, 
            examples=examples, 
            tokenizer=self.tokenizer)
        predictions = [predictions[i] for i in predictions.keys()]

        return list(zip(predictions, answer_scores))

# Generates an answer based on results/A list of SquadResult objects
def generate_answer_sent(results, features, examples, tokenizer):
    # results: A SquadResult abohect by Huggingface
    # features: The features generated by the specified tokenizer
    # examples: a list of squad examples
    # tokenizer: The preloaded tokenizer used together with the model

    predicted_outputs = compute_predictions_logits(
        all_examples=examples,
        all_features=features,
        all_results=results,
        n_best_size=2,
        max_answer_length=32,
        do_lower_case=True,
        output_prediction_file=False,
        output_nbest_file=False,
        output_null_log_odds_file=False,
        verbose_logging=False,
        version_2_with_negative=True,
        null_score_diff_threshold=0.0,
        tokenizer=tokenizer)

    if predicted_outputs == '':
        predicted_outputs = '[NO ANSWER]'

    return predicted_outputs

class score_answers:

    def __init__(self, start_scores, end_scores):
        # start_scores: start_logits generated by the question answering model
        # end_scores: end_logits generated by the question answering model

        self.start_scores = start_scores
        self.end_scores = end_scores

    # Runs the scoring prosedures
    # 
    def scores(self, seqlen, n_best=1):
        # Normalizes
        start_scores = self.normalize(self.start_scores)
        end_scores = self.normalize(self.end_scores)

        if start_scores.ndim == 1:
            start_scores = start_scores[None]

        if end_scores.ndim == 1:
            end_scores = end_scores[None]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start_scores, -1), np.expand_dims(end_scores, 1))

        # Remove candidate with end_scores < start_scores and end_scores - start_scores > seqlen
        candidates = np.tril(np.triu(outer), seqlen - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if n_best == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < n_best:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, n_best)[0:n_best]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
        return candidates[0, start, end][0]

    # Normalizes the scores using this equation
    # e^x/Σe^x
    def normalize(self, scores):
        ex = np.exp(scores)
        return ex/sum(ex)

# Displays a probability distribution of each answer 
# generates and saves the image to a folder with a random id
# You can see this data in https://github.com/sorcely/answer-analysis
# The function then returns the path the image is saved on for further use
def display_score_diagram(all_tokens, start_scores, end_scores, folder_name):
    start_scores = start_scores[0]
    end_scores = end_scores[0]

    # Store the tokens and scores in a DataFrame. 
    # Each token will have two rows, one for its start score and one for its end
    # score. The 'marker' column will differentiate them. A little wacky, I know.
    scores = []
    for (i, token_label) in enumerate(all_tokens):
        # Add the token's start score as one row.
        scores.append({
            'token_label': token_label,
            'score': start_scores[i],
            'marker': 'start'})
    
        # Add  the token's end score as another row.
        scores.append({
            'token_label': token_label, 
            'score': end_scores[i],
            'marker': 'end'})
    
    df = pd.DataFrame(scores)

    # Creates and show the diagram
    # Draw a grouped barplot to show start and end scores for each word.
    # The 'hue' parameter is where we tell it which datapoints belong to which
    # of the two series.
    g = sns.catplot(
        x='token_label', y='score', 
        hue='marker', data=df,
        kind='bar', height=6, aspect=4)

    # Turn the xlabels vertical.
    g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha='center')

    # Turn on the vertical grid to help align words to scores.
    g.ax.grid(True)

    plt.title('Word Scores')

    # Saves the plot to a random generated name
    os.makedirs(f'{ANSWER_ENGINE_PATH}/answer analysis/data/{folder_name}')
    plt.savefig(f'{ANSWER_ENGINE_PATH}/answer analysis/data/{folder_name}/answer-score.png')
