# New
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
from transformers.data.processors.squad import (
    SquadExample,
    SquadFeatures,
    SquadResult,
    SquadV2Processor)

# Multiprocessing
import threading
from threading import Thread

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

# Import typing
from typing import *
from transformers import PreTrainedModel, PreTrainedTokenizer

# Initialize global paths
ANSWER_ENGINE_PATH = 'answer_engine/' if __name__ != '__main__' else '' # Changing the path to files in the answer_engine folder
MODEL_PATH = '' if __name__ != '__main__' else '../' # Changing the path to files in the model folder

# Answers a question based on context and a question with the specified model type
class question_answering:

    def __init__(self, model:PreTrainedModel, tokenizer:PreTrainedTokenizer):
        '''
        Creates and returns a function to run the Question Answering AI

        ### Args ###
        model (:obj: `transformers.PreTrainedModel`)
            * Question answering model ready for inference
        tokenizer (:obj: `PreTrainedTokenizer`)
            * Tokenizer ready for inference
        '''

        self.model = model
        self.tokenizer = tokenizer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_batch_sz = 8

    # Runs the the prediction
    def __call__(self, contexts:Iterable[str], question:str, max_len:int = 384) -> Iterable[Union['prediction','score']]:
        '''
        Creates and returns a function to run the Question Answering AI

        ### Args ###
        contexts (:obj: `list`)
            An iterable of the contexts possibly containing the answer to the question
        question (:obj: `str`)
            The question we want answered
        max_len (:obj: `int`)
            The maximum length of the contexts
        '''

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
        feature_dict, features = squad_convert_examples_to_features_(
            examples=examples,
            tokenizer=self.tokenizer, 
            max_seq_length=max_len,
            doc_stride=128,
            max_query_length=64,
            threads=len(examples))

        # Sampels and converts the dataset to a dataloader
        batch_size = len(examples) if len(examples) < self.max_batch_sz else self.max_batch_sz
        dataloader = DataLoader(feature_dict, batch_size=batch_size)

        # Data variables
        results = []
        answer_scores = []

        # Sets the model into prediction mode
        self.model.eval()

        for batch in dataloader:
            # Set the inputs into the specified device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)

            with torch.no_grad():
                inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids}

                # The indices where the context is
                example_indices = batch['example_index']

                # Make a prediction
                outputs = self.model(**inputs)

                for i, examples_idx in enumerate(example_indices):
                    i_feature = feature_dict[examples_idx.item()]
                    unique_id = i_feature.unique_id

                    start_logits = outputs[0][i].detach().cpu().numpy()
                    end_logits = outputs[1][i].detach().cpu().numpy()

                    scores = score_answers(
                        start_scores = start_logits,
                        end_scores = end_logits,
                        seqlen=max_len,
                        n_best=1)
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

def squad_convert_examples_to_features_(examples:Iterable[SquadExample], tokenizer:PreTrainedTokenizer, max_seq_length:int, doc_stride:int, max_query_length:int, threads:int):
    '''
    Encode the examples into features
    '''

    args = {
        'max_seq_length': max_seq_length,
        'doc_stride': doc_stride,
        'max_query_length': max_query_length}

    def encode_fn(example:SquadExample, args:Dict, id_:int) -> Dict:
        context = example.context_text
        question = example.question_text

        output_dict =  tokenizer.encode_plus(
            text = question,
            text_pair = context,
            add_special_tokens = True,
            padding = 'max_length',
            truncation = 'only_second',
            max_length = args['max_seq_length'],
            stride = args['doc_stride'],
            return_tensors = 'pt')

        # Remove the "first" list so that it has the shape of [max_len]
        output_dict['input_ids'] = torch.squeeze(output_dict['input_ids'])
        output_dict['attention_mask'] = torch.squeeze(output_dict['attention_mask'])
        output_dict['token_type_ids'] = torch.squeeze(output_dict['token_type_ids'])

        # Add example_index: Represents what index this lies on
        output_dict.update({'example_index': id_})

        # unique id: Represents the specific feature
        output_dict.update({'unique_id': 1000000000 + id_})

        # qas_id: Represents the specific feature
        output_dict.update({'qas_id': example.qas_id})

        # token_to_orig_index: Represents the __XXX__
        # tokens: Represents each word in the context, so the word tokens in the context
        token_to_orig_index = []
        tokens = []
        for i, tok in enumerate(example.doc_tokens):
            sub_tokens = tokenizer.tokenize(tok)
            for sub_tok in sub_tokens:
                token_to_orig_index.append(i)
                tokens.append(sub_tok)

        # paragraph_len: Represents the length of the context tokens
        paragraph_len = len(token_to_orig_index)

        # token_to_orig_map: Represents where the given index of a token corresponds to in the given text (question + context)
        token_to_orig_map = {}
        sequence_added_tokens = tokenizer.max_len - tokenizer.max_len_single_sentence
        for i in range(paragraph_len):
            index = len(question.split(' ')) + sequence_added_tokens + i
            token_to_orig_map[index] = token_to_orig_index[i]

        # token_is_max_context: IDK
        token_is_max_context = {i:True for i in token_to_orig_map}

        # Create the features also
        features = SquadFeatures(
            input_ids = output_dict['input_ids'], # Required
            attention_mask = output_dict['attention_mask'], # Required
            token_type_ids = output_dict['token_type_ids'], # Required
            cls_index = None,
            p_mask = None,
            example_index = output_dict['example_index'], # Required
            unique_id = output_dict['unique_id'],         # Required
            paragraph_len = paragraph_len,                # Required
            token_is_max_context = token_is_max_context,  # Required 
            tokens = tokens,                              # Required
            token_to_orig_map = token_to_orig_map,        # Required
            start_position = None,
            end_position = None,
            is_impossible = False, 
            qas_id = output_dict['qas_id'])               # Required

        return output_dict, features

    # Code for multiprocessing (threads)
    class ThreadWithReturn(Thread):
        '''
        A custom Thread that actually returns a value
        We're using this Thread object to run the crawl and translate pages synchronously
        '''

        def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
            Thread.__init__(self, group=group, target=target, name=name, args=args, kwargs=kwargs)
            self._return = None

        def run(self):
            if self._target != None:
                self._return = self._target(*self._args, **self._kwargs)

        def join(self, *args):
            Thread.join(self, *args)
            return self._return

    # Spawns processes
    processes = []
    for i, ex in enumerate(examples):
        thread = ThreadWithReturn(
            target = encode_fn,
            args = (ex, args, i))
        thread.start()
        processes.append(thread)

    # Get the data from the processes
    feature_dict = []
    features = []
    for p in processes:
        p_result = p.join()
        if p_result:
            feature_dict.append(p_result[0])
            features.append(p_result[1])

    return feature_dict, features

# Generates an answer based on results/A list of SquadResult objects
def generate_answer_sent(results:Iterable[SquadResult], features:Iterable[SquadFeatures], examples:SquadExample, tokenizer:PreTrainedTokenizer) -> str:
    '''
    Creates and returns a function to run the Question Answering AI

    ### Args ###
    results (:obj: `SquadResult`)
        * A SquadResult object by Huggingface
    features (:obj: `SquadFeatures`)
        * The features generated by the specified tokenizer
    examples (:obj: `SquadExample`)
        * a list of squad examples
    tokenizer (:obj: `PreTrainedTokenizer`)
        * The preloaded tokenizer used together with the model
    '''

    predicted_outputs = compute_predictions_logits(
        all_examples=examples,
        all_features=features,
        all_results=results,
        n_best_size=2,
        max_answer_length=48,
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

def score_answers(start_scores:np.array, end_scores:np.array, seqlen:int, n_best:int = 1):
    '''
    Creates a score inspired by Chen & al. (https://github.com/facebookresearch/DrQA)

    ### Args ###
    start_scores (:obj: `np.array`)
        * start_logits generated by the question answering model
    end_scores (:obj: `np.array`)
        * end_logits generated by the question answering model
    '''

    # Normalizes the scores using this equation
    # e^x/Σe^x
    def normalize(scores:np.array):
        ex = np.exp(scores)
        return ex/sum(ex)

    # Normalizes
    start_scores = normalize(start_scores)
    end_scores = normalize(end_scores)

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

def display_score_diagram(all_tokens:np.array, start_scores:np.array, end_scores:np.array, folder_name:str):
    '''
    Displays a probability distribution of each answer generates and saves the image to a folder with a random id
    You can see this data in https://github.com/sorcely/answer-analysis

    ### Args ###
    all_tokens (:obj: `np.array`)
        * All the scores generated by the model
    start_scores (:obj: `np.array`)
        * start_logits generated by the question answering model
    end_scores (:obj: `np.array`)
        * end_logits generated by the question answering model
    folder_name (:obj: `str`)
        * The name of the folder where the image and json data will be saved
    '''
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
