import unittest
import requests

import sys
sys.path.append('../')

import data_engine

# Tests the run function
class DataEngineTests(unittest.TestCase):

    def test_run(self):
        '''
        Test the combined functions
        '''
        results = data_engine.run(
            query = 'donald trump election day',
            question = 'when was donald trump elected',
            n_results = 5,
            search_method = 'google',
            n_words = 256,
            lang = 'en')

        self.assertTrue(
            False not in [len(i) <= 256 for i in results['data']], 
            'The results is longer than ´n_words´')

    def test_google_search(self):
        '''
        In this test, we should only test if it compiles
        '''
        search_engine = data_engine.Search()
        results = search_engine('When did the WWII break out', 5, 'google')

    def test_news_api_search(self):
        # search_engine = data_engine.Search('<apikey>')
        # results = search_engine('When did the WWII break out', 5, 'newsapi')
        pass

    def test_webcrawler(self):
        '''
        In this test, we test if the crawler is able to:
          a) open the link, and behave correct if it's invalid
          b) Parse the text from the website
          c) the correct information
        But also if the output format is correct
        Is the word length long enough, and is it an iterable
        '''

        link = 'https://docs.python.org/3/library/typing.html'
        link1 = 'https://www.dr.dk/sporten/webfeature/gaetsportsgren'
        link2  = 'https://www.dr.dk/nyheder/viden/natur/forskere-maler-oejne-paa-koeers-rumper-det-holder-loeverne-vaek'
        
        results = data_engine.Webcrawler(
            urls = [link, link1, link2],
            n_results = 1)

        self.assertTrue(
            type(results) == list or type(results) == tuple)

    def test_workrank(self):
        data = [
                '''Køen strækker sig langt ned ad gaden foran Old Irish Pub i Odense.
                Blandt dem, der venter på at komme ind til høj musik, drinks og fadøl, er Sara Salomonsen, der lige er begyndt på sygeplejestudiet på UCL i Odense.
                Det er en af de uddannelsesinstitutioner, der er blevet ramt af et stort smitteudbrud og derfor har aflyst introdagene.
                En bytur med dem, studietiden skal deles med, tjener derfor et større formål.
                Vi har valgt at tage ud i byen for at skabe et fællesskab sammen, og det er svært at gøre i et klasselokale.
                I dag er vi ti piger, og planen er, at vi holder os sammen hele aftenen. Intentionen er ikke at tale med andre, men at lære hinanden at kende, siger hun.
                Vi vil egentlig bare gerne lave noget sammen på vores hold, lyder det fra hendes medstuderende Ditte Frisgaard Kristensen, inden de to opgiver at stå i den lange kø og i stedet bevæger sig mod et andet, mindre proppet værtshus.''',
                '''Og i går løftede sundhedsminister Magnus Heunicke pegefingeren på et pressemøde og advarede om, at nattelivet kan blive pålagt yderligere restriktioner, hvis det vurderes, at det er katalysator for smitte.
                At Mie Frederiksen befinder sig i et såkaldt corona-hotspot har dog ikke fået hende til at overveje at blive hjemme.
                Jeg har tænkt lidt over, at der er mange smittede, men der er mange smittede næsten alle steder, så hvorfor ikke bare tage af sted. Jeg passer jo på mig selv, siger hun.''']
        question = 'man venter på at komme ind'
        n_words = 256

        data = data_engine.bm25_okapi(
            *data,
            question = question,
            n_words = n_words)

if __name__ == '__main__':
    unittest.main()