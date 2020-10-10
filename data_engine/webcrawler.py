from bs4 import BeautifulSoup as Soup
from bs4 import Comment
import requests

import googletrans
translator = googletrans.Translator()

import threading
from threading import Thread

from typing import Iterable, Callable, Optional

import httpcore

def Webcrawler(urls:Iterable, n_results:int) -> Iterable:
    '''
    Opens and crawls the given list of urls and then returns 
    a list of webpage content with the range of n_results

    urls (:obj: 'Iterable')
        * The urls that we want to extract the information
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

        ### Args ###
        inputs (:obj: 'dict')
            * A dict containing inputs the functions
            * It contains the url
            * url contains the textual data that we want to extract
        '''

        url = inputs['url']

        # Run the crawling function
        text = Crawl(url)

        # Translate the text
        if text:
            try:
                text = translator.translate(text, dest='en').text
            except httpcore._exceptions.WriteError as err:
                print('**Could not translate data. Continuing...')

        return text

    # Iterate over each link to spawn a process
    processes = []
    for u in urls:
        thread = ThreadWithReturn(
            target=MultiThreadProcess, 
            args=[{'url': u}])
        thread.start()
        processes.append(thread)

    # Extract the data from the processes
    data = []
    for p in processes:
        p_result = p.join()
        if p_result:
            data.append(p_result)

    return data

def Crawl(url:str) -> str:
    '''
    The function that opens, crawls and parses the given url.
    It's used in the multiprocess.

    ### Args ###
    url (:obj: 'str'): 
        The url we want to open
    '''

    # Get the page source
    try:
        res = requests.get(url, timeout=0.5)
        timeout_err = False
    except requests.exceptions.Timeout as err:
        timeout_err = True

    # Checks if the connection was successfull
    if not timeout_err:
        if res.status_code == 200:
            content = res.content

            # Close connections
            res.close()

            # Parse the html using BeautifulSoup
            parsed_content = Soup(content, 'html.parser')
            
            # Filter out script, style, comments tags
            if parsed_content.script != None:
                _ = [i.decompose() for i in parsed_content.find_all('script')]
            if parsed_content.style != None:
                _ = [i.decompose() for i in parsed_content.find_all('style')]
            if parsed_content.script != None:
                _ = [i.decompose() for i in parsed_content.find_all(text=lambda t: isinstance(t, Comment))]
            del _

            paragraphs = parsed_content.find_all(text=True)

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
    results = Webcrawler(
        urls = [
            'https://news.usc.edu/86362/fukushima-disaster-was-preventable-new-study-finds/',
            'https://news.usc.edu/trojan-family/mars-colony-on-earth-hawaii-usc-alumni/'],
        n_results = 256)
    print(results[0])
