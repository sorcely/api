import unittest
import requests

import sys
sys.path.append('../')

import data_engine

# Tests the run function
class DataEngineTests(unittest.TestCase):

    def test_google_search(self):
        '''
        In this test, we should only test if it compiles
        '''
        search_engine = data_engine.Search()
        results = search_engine('When did the WWII break out', 5, 'google')

    def test_news_api_search(self):
        pass

    def test_webcrawler(self):
        '''
        In this test, we test if the crawler is able to:
         a) open the link, and behave correct if it's invalid
         b) Parse the text from the website
         c) Return the correct information
        But also if the output format is correct
        Is the word length long enough, and is it an iterable
        '''
        link = 'https://docs.python.org/3/library/typing.html'
        link1 = 'https://www.dr.dk/sporten/webfeature/gaetsportsgren'
        link2  = 'https://www.dr.dk/nyheder/viden/natur/forskere-maler-oejne-paa-koeers-rumper-det-holder-loeverne-vaek'
        results = data_engine.Webcrawler(
            urls = [link],
            question = 'what does this provide',
            n_results = 1,
            n_words = 256)

        self.assertTrue(type(results) == list)
        self.assertTrue(len(results[0].split(' ')) <= 256)

if __name__ == '__main__':
    unittest.main()