from rank_bm25 import BM25Okapi

import numpy as np
import math as M
import nltk
from nltk import word_tokenize as tokenize

import re
import string
punctuation = set(string.punctuation)
punctuation.remove('.')
regex = re.compile('[%s]' % re.escape(string.punctuation))

from typing import Iterable, Union

def bm25_okapi(*texts:Iterable, question:str, n_words:int = 256) -> Iterable[str]:
    '''
    Returns a lists of strings with sentences in descending order, ordered by the scores.
    We're using the rank_bm25 module to easily integrate the famous BM25 algorithm
    The sentences is given the scored based on the question. In a future version maybe 
    include word vectors, so that the meaning of each word is scored rather than the word itself

    ### Args ###
    texts (:obj:'iterable')
        * A list of texts extracted from the websites
    question (:obj:'str')
        * The question asked by the user
        * Using the question instead of the query, since we want to get the sentence most likely to get answered
    n_words (:obj:'int')
        * The amount of words each text output should be
        * Smart to have the same as our QA and classification model
    '''

    sentences = []
    for text in texts:
        # Remove punctuation
        text = regex.sub('', text)
        text = text.replace('. ', '.')
        text = text.split('.')

        # Tokenize the  the texts
        # Maybe change this to a 
        text_ = [tokenize(i) for i in text]

        question_ = tokenize(question)

        # Initialize the bm25 okapi module
        bm25_fn = BM25Okapi(text_)
        
        # Run the scores
        scores = bm25_fn.get_scores(question_)

        # Rearrange the text
        # Merge scores and sentences (original context)
        scored_sentences = list(zip(text, scores))

        # Sort the sentences in an ascending order (high -> low)
        sorted_sentences = sorted(scored_sentences, key=lambda sentence: sentence[1], reverse=True)

        sentence = ''
        words = 0
        for i,_ in sorted_sentences:
            words += len(i.split(' '))
            if words < n_words:
                sentence += i
                continue
            break
    
        sentences.append(sentence)

    return sentences

def BM25_custom(*texts:Iterable, secondary_text:str, batch:bool = False) -> Iterable[str]:
    '''
    A custom written (still unfinished) BM25 algorithm
    The reason for writing this, is to quickly develop word vectors into it.
    But currently it's planned to just keep it as a stand word-token algorithm

    ### Args ###
    texts (:obj:'iterable')
        * A list of texts extracted from the websites
    secondary_text (:obj:'str')
        * Basically the question asked by the user
        * Using the question instead of the query, since we want to get the sentence most likely to get answered
    batch (:obj:'bool')
        * Whether to compare each text in batches. So it also takes into consideration the other texts.
    '''

    def parse(corpus:str) -> Union[dict, list]:
        '''
        Format the corpus and create a dict of word frequencies
        '''
        
        term_freqs = {}
        term_tokens = []

        for w in corpus.split(' '):
            # Add to the tokenized sentence
            term_tokens.append(w)
            
            # Remove punctuation
            w = regex.sub('', w)

            # Add to dict if not seen before
            if not w in term_freqs:
                term_freqs[w] = 0

            # Give the word a score 
            term_freqs[w] += 1
        
        return term_freqs, term_tokens

    def inverse_term_freq(corpus_len:int, contains_n:int) -> idk:
        '''
        Calculates inverse frequencies of terms in the given corpus
        It's also known as just IDF

        ### Args ###
        corpus_len (:obj: 'int')
        contains_n (:obj: 'int')
        '''

        return M.log(
            (corpus_len - contains_n + 0.5) / (contains_n + 0.5))

    def term_freq(term:str, term_freqs:dict) -> float:
        '''
        Using tge term frequency algorithm
        f(t,d) / Σ f(t',d)

        ### Args ###
        term
        term_freqs
        '''

        # Term frequency of the given term
        tf = term_freqs[term]

        # The sum of all term frequencies
        tf_all = term_freq_sum = sum(
            [term_freq[i] for i in term_freq])

        return tf / (tf_all - tf)

    def scoring(doc_len:int, avg_doc_len:float, k1:float, b:float):
        '''
        Calculates the scores

        ### Args ###
        doc_len (:obj: 'int')
        avg_doc_len (:obj: 'float')
        avg_doc_len (:obj: 'float')
        avg_doc_len (:obj: 'float')
        '''

        for i in term_tokens:
            pass    

        return None

    def batch_scoring():
        return None

    return None
