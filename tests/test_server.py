import unittest
import requests
import json

import sys
sys.path.append('../')

import server

class WebServerTests(unittest.TestCase):

    # Sets up the app before tests
    def setUp(self): 
        # Starts the app
        server.app.run(
            debug=True,
            host='127.0.0.1',
            port='5000')

    # Tears down the app after tests
    def tearDown(self):
        pass

    # Tests that the main views works by making an request
    def test_main_api(self):
        url = '127.0.0.1:5000/'
        headers = '?question=Did donald trump kill osama bin laden'
        req = requests.get(url + headers)

        # Tests if the response is valid
        self.assertEqual(req.status_code, 200)

        # Tests if the output is formatted correctly
        out = json.loads(req.text)

        # Tests if the keys is in the output
        self.assertTrue('question' in out)
        self.assertTrue('results' in out)

        # Tests if the data is in the results
        self.assertTrue('links' in out['results'])
        self.assertTrue('answers' in out['results'])
        self.assertTrue('answer_scores' in out['results'])
        self.assertTrue('article_type' in out['results'])

if __name__ == '__main__':
    unittest.main()