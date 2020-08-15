import requests
import json

# Gets the links from the query/search
def get_links(query, date, sort_by, api_key):
    # query: the keywords from the question/search
    # date: the date to sort articles from
    # sort_by: how to do the sorting of articles
    # api_key: the key to api. It exists in an enviroment variable

    url = f'http://newsapi.org/v2/everything?q={query}&from={date}&sortBy={sort_by}&apiKey={api_key}'
    response = requests.get(url)

    # Convert response to python datatypes from json
    response = json.dumps(response)

    # The data contains only urls for now
    # but we should probably do some reasoning
    # basically choosing the right article by saving description
    data = []

    # Iterate over articles
    if response['status'] != 'error':
        for article in response['articles']:
            # Find article links
            url = article['url']
            data.append(url)
    else: return None

    return data
