import search_engine_parser
from search_engine_parser import (
    GoogleSearch, 
    BingSearch,
    DuckDuckGoSearch)

import newsapi
from newsapi.newsapi_client import NewsApiClient

import requests
import json
from typing import Iterable, Callable, Optional

BUFFER = 5

# Search object for gathering links to further analysis
# It uses class architechture, to among other things, save search_methods 
class Search:

    def __init__(self, newsapi_key:Optional[str] = None):
        '''
        ### Args ###
        newsapi_key (:obj: 'str')
            * The api key to gain access to the news api
            * Right now it's stored in the enviroment variable
        '''

        # A list of all available search_engines
        # The first link will always be the standard
        self.search_methods = [
            'google',
            'bing',
            'duckduckgo',
            'newsapi']

        if newsapi_key:
            self.newsapi_obj = newsapi.NewsApiClient(newsapi_key)
        else:
            self.search_methods.remove('newsapi')

    # This makes the actual process
    def __call__(self, query:str, n_links:int, search_method:str = 'google', lang:str = None) -> Iterable:
        '''
        Args:
            query (:obj: 'str')
               * The search query we are passing to the specified search engine
               * It can be different from the question. Maybe the user wants so ask a totally 
                different question than the search querybut as standard it is the question
            n_links (:obj: 'int')
               * The maximum amount of links that is returned
            search_method (:obj: 'str')
               * It's the type of search engine we want to use. Maybe Google, NewsAPI or another one
               * This can be effective if Google are IP banning you/us. But also if we want 
                to search news articles
               * We are validating the search_method through the search_methods function
            language (:obj: 'str')
               * It must be a ISO-638-1
               * Langauge of the question and thereby also the query
               * Used to get specific articles from the specified country
        '''

        # Validate search_method
        search_method = self.check_search_method(search_method)

        # Load search engine function
        search_engine = self.create_search_fn(search_method)

        # 1.Search for query
        if search_method == 'newsapi':
            results = search_engine(query=query, lang=lang)
        else:
            results = search_engine(query=query)

        # Take only the links
        results = results['links']

        # Take max n_links
        return results[:n_links+BUFFER]

    def create_search_fn(self, method:str) -> Callable:
        '''
        Args:
            method (:obj: 'str')
               * The name of the way we want to search
               * This could be 'google', 'newsapi'
        '''

        if method == 'google':
            def search_fn(query):
                args = (query, 1)
                results = GoogleSearch().search(*args)
                return results
            return search_fn

        elif method == 'newsapi':
            def search_fn(query, lang):
                results = newsapi_fn(
                    api_obj=self.newsapi_obj,
                    query=query,
                    lang=lang)
                # Turn the results into a dict
                results = {'urls': results}
                return results
            return search_fn

        elif method == 'bing':
            def search_fn(query):
                args = (query, 1)
                results = BingSearch().search(*args)
                return results
            return search_fn

        elif method == 'duckduckgo':
            def search_fn(query):
                args = (query, 1)
                results = DuckDuckGoSearch().search(*args)
                return results
            return search_fn

    def check_search_method(self, method:str) -> str:
        '''
        Args:
            method (:obj: 'str')
               * The name of the search method we want to use
               * That string is then checked is correct in this function
        '''

        method = method.lower()        

        if method in self.search_methods:
            return method

        return self.search_methods[0]

def newsapi_fn(query:str, lang:str = 'en', api_obj:NewsApiClient = None) -> Iterable:
    '''
    Args:
        api_obj (:obj: 'NewsApiClient', :default: 'None')
           * A preinitialized client api object
           * We're loading it before this func, to save a little bit of time, 
            and it would be pointless to do it over and over again
        query (:obj: 'str')
           * The search query specified by the user
           * This function is the main reason for splitting question and query
            because this will achieve better results with the just the headline
        lang (:obj: 'str', :default: 'en [english]')
           * That's the langauge we want to search in
           * It will be automatically generated by the google translate api
            It returns the original language which is pretty cool
    '''

    # Send a request to the API
    response = api_obj.get_everything(
        q=query,
        language=lang,
        sort_by='relevancy')

    # Convert from json to dict
    response = json.loads(response)

    # Only get links if we successfully sent the request
    if response['status'] != 'error':
        # Save urls
        #! In a future version, probably do some logic of the description, so we get the most accurate news articles
        urls = []
        for article in response['articles']:
            url = article['url']
            urls.append(url)
        return urls
    return None

if __name__ == "__main__":
    Search_engine = Search()
    r = Search_engine('When did the WWII break out', 5, 'google')
    [print(i) for i in r]
    print('len r:',len(r))
