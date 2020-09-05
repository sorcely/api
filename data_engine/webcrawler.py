from bs4 import BeautifulSoup as Soup
import requests

import googletrans
translator = googletrans.Translator()

import threading
from threading import Thread

from typing import Iterable, Callable, Optional

def Webcrawler(urls:Iterable, question:str, n_results:int) -> Iterable:
    '''
    asdsdasdsadsadsdas
    asdsdasd asd
    asd asd asdsadasdasd ad

    Args:
        urls (:obj: 'Iterable')
           * The urls that we want to extract the information
        question (:obj: 'str')
           * The question asked by the user
           * Used to rank the sentence with the highest 
            probability of finding the answer within it
        n_results (:obj: 'int')
           * The maximum results we want to get
           * We're doing this because we also have the buffer
            And we don't want more results than specified
    '''

    def MultiThreadProcess(inputs:dict) -> str:
        '''
        The process that is spawned in the specified amount of threads
        We're using threading since we aren't really calculating (beside the word rank)
        But all other function are using API's or just "downloading" stuff

        Args:
            inputs (:obj: 'dict')
               * A dict containing inputs the functions
               * It contains the url, question
               * url contains the textual data that we want to extract
               * question is the question asked by the user, and is used to find the most probable sentences
        '''

        url = inputs['url']
        question = inputs['question']

        # Run the crawling function
        text = Crawl(url)

        # Translate into english
        # text = translator.translate(text, dest='en').text

        return text

    # Iterate over each link to spawn a process
    processes = []
    for u in urls:
        thread = ThreadWithReturn(
            target=MultiThreadProcess, 
            args=[{'url': u, 'question': question}])
        thread.start()
        processes.append(thread)

    # Extract the data from the processes
    data = []
    for p in processes:
        p_result = p.join()
        # If results != None
        if p_result:
            data.append(p_result)

    return data

def Crawl(url:str) -> str:
    '''
    The function that opens, crawls and parses the given url
    It's used in the multiprocess

    Args:
        url (:obj: 'str'): The url we want to open
    '''

    # Get the page source
    res = requests.get(url)

    # Checks if the connection was successfull
    if res.status_code == 200:
        content = res.content
            
        # Close connection
        res.close()

        # Parse the html using BeautifulSoup
        parsed = Soup(content, 'html.parser')
        paragraphs = parsed.find_all(text=True)

        # Turn the paragraphs into a string
        paragraphs = ' '.join(paragraphs)

        return paragraphs
    return None

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

if __name__ == "__main__":
    Webcrawler(
        ['https://docs.python.org/3/library/typing.html',
         'https://news.usc.edu/86362/fukushima-disaster-was-preventable-new-study-finds/'],
        'Heysa, what\'s your name',
        2,
        256)