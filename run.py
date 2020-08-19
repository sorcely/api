import answer_engine
import data_engine
import googletrans

import googletrans

import os
import json
import concurrent.futures
import threading
from threading import Thread

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def remove_bad_chars(x):
    # Where x is a string
    # remove_bad_chars is used to replce weird unseen character with the closest character
    bad_chars = {
        '¢': 'c', '¥': 'Y', 'Á': 'A', 'Â': 'A', 'É': 'E', 'Î': 'I', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O', '×': 'x', 'Ú': 'U', 
        'Ü': 'U', 'ß': 'ss', 'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'ç': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e', 
        'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i', 'ð': 'o', 'ñ': 'n', 'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ù': 'u', 
        'ú': 'u', 'û': 'u', 'ü': 'u', 'ý': 'y', 'ÿ': 'y', 'Ā': 'A', 'ā': 'a', 'ă': 'a', 'ć': 'c', 'č': 'c', 'Đ': 'D', 'đ': 'd', 
        'Ē': 'E', 'ē': 'e', 'ė': 'e', 'ę': 'e', 'ě': 'e', 'ğ': 'g', 'ħ': 'h', 'ī': 'i', 'İ': 'I', 'ı': 'l', 'ļ': 'l', 'Ł': 'L', 
        'ł': 'l', 'ń': 'n', 'ņ': 'n', 'ŋ': 'n', 'Ō': 'O', 'ō': 'o', 'ő': 'o', 'œ': 'æ', 'ś': 's', 'Ş': 'S', 'ş': 's', 'Š': 'S', 
        'š': 's', 'ũ': 'u', 'ū': 'u', 'ů': 'u', 'ŵ': 'w', 'Ż': 'Z', 'ż': 'z', 'Ž': 'Z', 'ž': 'z', 'ư': 'u', 'ǎ': 'a', 'ǐ': 'i', 
        'ǒ': 'o', 'ǔ': 'u', 'ș': 's', 'ț': 't', 'ɐ': 'a', 'ɑ': 'a', 'ɒ': 'a', 'ɔ': 'c', 'ə': 'e', 'ɛ': '3', 'ɜ': '3', 'ɡ': 'g', 
        'ɣ': 'Y', 'ɫ': 'l', 'ɭ': 'l', 'ɯ': 'w', 'ɳ': 'n', 'ɸ': 'ø', 'ɹ': 'j', 'ʂ': 's', 'ʏ': 'y', 'ʒ': '3', 'ʰ': 'h', 'Ε': 'E', 
        'Η': 'H', 'Θ': 'O', 'Ι': 'I', 'Κ': 'K', 'Λ': 'A', 'Μ': 'M', 'Ν': 'N', 'ί': 'i', 'α': 'a', 'β': 'B', 'γ': 'Y', 'η': 'n', 
        'θ': '0', 'ν': 'v', 'ώ': 'w', 'З': '3', 'И': 'N', 'К': 'K', 'Н': 'H', 'П': 'n', 'С': 'C', 'Т': 'T', 'а': 'a', 'б': '6', 
        'в': 'B', 'д': 'A', 'е': 'e', 'з': '3', 'и': 'n', 'й': 'n', 'к': 'k', 'н': 'H', 'о': 'o', 'п': 'n', 'р': 'p', 'с': 'c', 
        'т': 'T', 'у': 'y', 'ч': '4', 'ш': 'w', 'ь': 'b', 'я': 'R', 'ᵻ': 'l', 'ḍ': 'd', 'Ḥ': 'H', 'ḥ': 'h', 'ḷ': 'l', 'ṃ': 'm', 
        'ṅ': 'n', 'ṇ': 'n', 'ṭ': 't', 'ẓ': 'z', 'ả': 'a', 'ấ': 'a', 'ế': 'e', 'ề': 'e', 'ễ': 'e', 'ỉ': 'i', 'ồ': 'o', 'ớ': 'o', 
        'ử': 'u', 'ữ': 'u', 'ἀ': 'a', 'ἁ': 'a', 'ἄ': 'a', 'Ἀ': 'A', 'Ἄ': 'A', 'Ἑ': 'E', 'ἡ': 'n', 'Ἥ': 'H', 'ἰ': 'i', 'ἴ': 'i', 
        'ἶ': 'i', 'Ἰ': 'I', 'ὐ': 'u', 'ὑ': 'u', 'ῖ': 'i', 'ῦ': 'u', 'ῶ': 'w', 'ﬁ': 'fi', 'ﬂ': 'fl'}

    for c in bad_chars:
        x = x.replace(c, bad_chars[c]) if c in x else x

    return x

# Checks whether a given text is fake news
# It returns the answer and how fake news 
# like the article is written
class fakenews:

    # Initialize the modules used to check for fake news
    def __init__(self, search_engine:str, threads:int=8):
        # Search_engine: The standard way of getting information

        # Threads: Is basically the amount of websites to scrape
        self.threads = threads

        # Initialize the google object
        self.translator = googletrans.Translator()

        # Initialize the data engine module
        # ...

        # Initalize the answer_engine module
        self.pipeline = answer_engine.pipeline(
            model_names={
                'qa': 'ktrapeznikov/albert-xlarge-v2-squad-v2',
                'cs': 'GRU-article-crediability.h5'},
            use_custom_pipeline=False, 
            from_google_bucket=False)

    def check(self):
        # n_word: the max amount of words the webcrawler outputs
        n_words = 256

        if len(data) > 0:
            # Translates the query into plain english
            query = self.translator.translate(query, dest='en').text

            # Do question answering
            answers = self.pipeline(
                contexts=data, 
                question=query)

            answers, answer_scores, crediabilities = list(zip(*answers))

            return {
                'links': links,
                'answers': answers,
                'answer_scores': answer_scores,
                'crediabilities': crediabilities}

        return {'error': 'No results - for some reason we didn\'t find any results with your query'}

if __name__ == '__main__':
    m = fakenews('google', threads=2)
    r = m.check('Where does covid19 originate from')
    print(r)
