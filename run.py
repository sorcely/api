import answer_engine
import data_engine
import googletrans

import googletrans

import os
import json
import concurrent.futures
import threading
from threading import Thread

from typing import Iterable, Optional, Dict

# Checks whether a given text is fake news
# It returns the answer and how fake news 
# like the article is written
class fakenews:

    # Initialize the modules used to check for fake news
    def __init__(self):
        # Initialize the google object
        self.translator = googletrans.Translator()

        # Initalize the answer_engine module
        self.pipeline = answer_engine.pipeline(
            model_names={
                'qa': 'ktrapeznikov/albert-xlarge-v2-squad-v2',
                'cs': 'GRU-article-crediability.h5'},
            use_custom_pipeline=False, 
            from_google_bucket=False)

    def check(self, question:str, n_results:int, n_words:int, search_method:str, lang:str, query:Optional[str] = None) -> dict:
        '''
        Runs the fact check function
        a) Gather data using the data_engine
            1) Make a search on the specified method
            2) Open each url and extract the text from the webpage
            3) Minimize the text, to the most important sentences
        b) Answer the given question
            1) Preprocess data using specified tokenizer
            2) Input into the model
            3) Generate answers and scores
        It then returns a list of dicts containing important information about the urls
        return Dict[urls, answers, answer_scores, biases]

        ### ARGS ###
        query (:obj: 'str')
            * Query a.k.a search query. This is the search term
            * As default it is the same as question
            * However it may be a good idea to seperate those
        question (:obj: 'str')
            * The question that is being asked
            * Used to rank each sentence and by that making the ML models faster
        n_results (:obj: 'int')
            * It's the results that is being returned
        n_words (:obj: 'int')
            * Maximum words in each data point
            * The higher the slower our ML-models will be, but i contains more information
        search_method (:obj: 'str')
            * See search.py and then Search.search_methods to get an idea of what search engines is included
        lang (:obj: 'str[ISO-638-1] language code')
            * It must be a ISO-638-1
            * Langauge of the question and thereby also the query
            * Used to get specific articles from the specified country
        '''

        # Set the query to question if not specified
        if query == None:
            query = question

        # Run the data_engine
        search_results = data_engine.run(
            query = query,
            question = question,
            n_results = n_results,
            n_words = n_words,
            search_method = search_method,
            lang = lang)

        data = search_results['data']
        urls = search_results['urls']

        # Checks if any data is returned
        if data:
            # Translates the query and question into plain english
            question_en = self.translator.translate(question, dest='en').text

            # Do question answering
            answers = self.pipeline(
                contexts=data, 
                question=question_en)

            answers, answer_scores, crediabilities = list(zip(*answers))

            return {
                'urls': urls,
                'answers': answers,
                'answer_scores': answer_scores,
                'biases': crediabilities}

        return {
            'error': '''
                No results - for some reason we didn\'t find any results with your query. 
                This may be a server/code problem. We ask you go open a issue on https://github.com/sorcely/api-lite/issues. 
                This can get the problem resolved and maybe even help others.'''}

if __name__ == '__main__':
    fakenews_checker = fakenews()

    results = fakenews_checker.check(
        question='Where does corona originate from',
        query=None,
        n_results=5,
        n_words=128,
        search_method='google',
        lang='en')

    print(results['answers'])
    print(results['answer_scores'])
    print(results['biases'])
