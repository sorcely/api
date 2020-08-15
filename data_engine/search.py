import requests

# The google search_engine parser
# This search engine is really slow and is not recommended for production
from search_engine_parser import GoogleSearch, DuckDuckGoSearch

class search_engine:

    def __init__(self, name):
        # name: the name of the search engine we want to use

        # Create search engine functions
        def bing_fn(query):
            # Config
            url = 'https://api.cognitive.microsoft.com/bing/v7.0/search'
            api_key = 'API KEY'

            headers = {"Ocp-Apim-Subscription-Key": api_key}
            params = {"q": query, "textDecorations": True, "textFormat": "HTML"}

            # Sends the request
            res = requests.get(
                url, 
                headers=headers, 
                params=params)
            res.raise_for_status()
            search_results = res.json()

            return search_results['webPages']

        def duckduckgo_fn(query):
            # Initialize the duckduckgo search engine
            search_args = (query, 1) # where 1 is the amount of pages to crawl
            dsearch = DuckDuckGoSearch()

            # Sends a request
            dresults = dsearch.search(*search_args)
            return dresults['links']

        def google_fn(query):
            # Initialize the duckduckgo search engine
            search_args = (query, 1)
            gsearch = GoogleSearch()

            # Sends a request
            gresults = gsearch.search(*search_args)
            return gresults['links']

        # Specifies the search engine to be used
        if name == 'bing':
            self.search_engine = bing_fn
        elif name == 'duckduckgo':
            self.search_engine = duckduckgo_fn
        else:
            self.search_engine = google_fn

    def search(self, query, n_links):
        # query: the search query which is also the question specified by the user
        # n_links: the amount of links to return

        # Makes the search engine
        # We don't reduce the result before we've filtered them
        search_results = self.search_engine(query)

        # Filters out links that often don't work
        search_results = self.filter_results(search_results)

        return search_results[:n_links]

    def filter_results(self, urls):
        non_valid = [
            '.xls', '.txt', 'youtube.com', 'youtu.be']

        # Filter results
        str_urls = ' '.join(urls)
        for i in non_valid:
            # Adds a filter so we don't 
            # always have to iterate over urls
            if i in str_urls:
                for j in urls:
                    if i in j:
                        urls.remove(j)

        return urls
