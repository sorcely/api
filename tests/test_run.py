import unittest

import sys
sys.path.append('../')

import run

# Tests the run function
# This include more advanced testing
class FakeNewsTests(unittest.TestCase):
    
    def setUp(self):
        self.fakenews = run.fakenews('google')

    def tearDown(self):
        pass

    def test_google(self):
        out = self.fakenews.check(query='Did osama bin laden create isis')

        if 'error' not in out:
            # Tests if the keys is in the output
            self.assertTrue('question' in out)
            self.assertTrue('results' in out)

            # Tests if the data is in the results
            self.assertTrue('links' in out['results'])
            self.assertTrue('answers' in out['results'])
            self.assertTrue('answer_scores' in out['results'])
            self.assertTrue('article_type' in out['results'])
        else:
            # Turns it into an error no matter what
            self.assertTrue(False)

    def test_advanced_search(self):
        pass
        # self.fakenews.check('Did osama bin laden create isis')

if __name__ == '__main__':
    unittest.main()