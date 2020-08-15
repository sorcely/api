from bs4 import BeautifulSoup as Soup
import requests
import re
import os
import sys

# Text ranking
import nltk
from nltk.corpus import stopwords
import rank_bm25 

# Import the search engine module
from .search import *

# We can't import it from run.py
# So we just copy the code into this too
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

# Save the urls which went wrong
# and also the reason it went wrong
def save_invalid_url(url, error):
    with open('invalid_urls.txt', 'a') as f:
        f.write(f'{url} --- {error} \n\n')

# Get all paragraphs from a website 
def webcrawler(url:str, question:str, max_words:int, use_newsapi:bool=False) -> str:
    # url: will be opened via requests and then scraped
    # question: question to compare each sentence to
    # max_words: the maximum words that will be outputted

    # We're using a try except block so that if the website 
    # isn't visible for at bot, we can just blacklist the link
    try:
        o = Soup(
            requests.get(url).text, # Opens the url and collects text
            'html.parser' # The parser used on the webpage
        ).find_all('p') # Finds all paragraph texts
    except Exception as e:
        save_invalid_url(url, e)
        return None

    # Saves only the text from the paragraphs
    o = remove_bad_chars([i.text for i in o])
    o = ' '.join(o)

    # Rank each sentence given
    o = word_rank(
        context=o,
        query=question,
        max_words=max_words)

    return o

def word_rank(context:str, query:str, max_words:int) -> str:
    # context: the data from the website
    # question: question to compare each sentence to
    # max_words: the maximum words that will be outputted

    # Tokenize words
    context = context.split('. ')

    # Tokenizes each sentence in the context and adds it to the list
    context_tokens = [nltk.word_tokenize(i) for i in context]

    # Tokenize the question
    # We don't need a list comprehension 
    # since this is just a single sentence
    query_tokens = nltk.word_tokenize(query)

    # Rank each sentence
    # Initializes the bm25-okapi module
    bm25 = rank_bm25.BM25Okapi(context_tokens)

    # Scores each sentence based on the question
    scores = bm25.get_scores(query_tokens)

    # Merge scores and sentences (oj context)
    scored_sentences = list(zip(context, scores))

    # Sort the sentences in an ascending order (high -> low)
    sorted_sentences = sorted(scored_sentences, key=lambda sentence: sentence[1], reverse=True)

    # Removes unnecessary sentences which exceeds the max words
    output = ''
    words_len = 0
    for i, _ in sorted_sentences:
        words_len += len(i.split(' '))
        if words_len >= max_words:
            break
        # Else append sentences to the output variable
        output += i + '. '

    return output
