import unittest
import requests

import sys
sys.path.append('../')

import data_engine

# Tests the run function
class DataEngineTests(unittest.TestCase):

    def test_google_search(self):
        g = data_engine.search_engine('google')
        r = g.search('What is unittesting', 10)

        self.assertTrue(len(r) <= 10) # Fail if r is above 11

    def test_webcrawler(self):
        working_link = 'https://www.dr.dk'
        not_working_link = 'https://www.dr.dk'
        question = ''

        # Open invalid_urls.txt so we can compare it with a future version
        with open('invalid_urls.txt', 'r') as f:
            f_original = f.read()

        r1 = data_engine.webcrawler(working_link, question, 256)
        r2 = data_engine.webcrawler(not_working_link, question, 256)

        # Open invalid_urls.txt so we can compare it with a future version
        with open('invalid_urls.txt', 'r') as f:
            f_new = f.read()

        # Assertions to the first request
        self.assertTrue(r1 != None)
        self.assertTrue(len(r1) <= 256)

        # Assertions to the not working url
        self.assertTrue(f_original == f_new)

if __name__ == '__main__':
    unittest.main()