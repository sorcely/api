import sys
sys.path.append('../data_engine/')
sys.path.append('data_engine/')
from search import *
from webcrawler import *
from wordrank import *

from typing import Iterable

def run(
	query:str,
    question:str,
    n_results:int,
    search_method:str = 'google',
    n_words:int,
    lang:str = 'en') -> Iterable:

    '''
    This function runs the data engine functions
    More specifically
	  a) Get links
	  b) Open urls
	  c) Rank each text
	  d) return an iterable

    Args:
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

    # Make a search to the specified search_method
    search_fn = Search(None)
    urls = search_fn(
        query = query, 
        n_links = n_results,
        search_method = search_method,
        lang = lang)

    # Crawl urls to extract information
    data = Webcrawler(
        urls = urls,
        question = question,
        n_results = n_results,)

    # Shrink the data
    bm25_okapi(
        *data,
        question = question,
        n_words = n_words)

    return data