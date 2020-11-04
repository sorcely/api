# Contributing to Sorcely Api codebase
👍🎉 First off, thanks for taking the time to contribute! 🎉👍

## Overview of the codebase
### AnswerEngine
The answer engine is supposed to provide the user meaningfull answers to the given question.  
We do that by using question answering system to derive answers from text given by the data engine.  

#### answering.py
In this script we create functionality to create the optional answer, data analysis tools and answer scoring.  
The code should be as easy as possible to use, meaning that we aim for creating single functions without any huge classes behind them.  

#### pipelines.py
In this script we combine all of the functionalties into an algorithm for finding, saving and scoring the answers generated.  
We currently we have to versions, An extended huggingface pipeline and our own pipeline. We plan to deprecate the Huggingface version due to the fact that it isn't as scalable and customizable as writing our own code. Ofcourse we still use the some of the functions such as logits decoding. However we want to have as small as possible components to work with.

### DataEngine
The data engine is probably the most important part of this whole code base since this is how we are able to get the best quality data. However this is also the less sexiest.  

#### __init__.py
init.py is only containing the run function. The run function runs all of the other functionality such as search, webcrawling and wordrank.  

#### search.py 
Creates the search methods. We use the search engine parser as the main underlaying technology. This helps us scrape search engines such as Google, DuckDuckGo, Bing and so on.
Currently we only support Google, NewsAPI, DuckDuckGo.  
In the future we want to add many different types of search engines to get the best types of data.

#### webcrawler.py
Responsible for scraping each website gathered by the search engine.  
This does not include NewsApi.  
Currently we only extract P tags. But we'd like to also add paragraphs longer than 20 words or so.  

#### wordrank.py
Wordrank chooses what sentences to extract from the webpages using the Okapi algorithm.  
This is a hard task to do, so in the future we want to put more focus onto this part of the engine.

### server.py
Here we serve the API on a virtual machine. Not much to say about this part.

### run.py
Combines all engines into one single class we can run for inference.  
There isn't much to say about this too other than we want to keep this section as free from new code function as possible. We'd prefer to classify the new function into one of the answer or data engine.

## Submitting changes
### Steps to implement a new feature
1. Submit an issue describing clearly what you want to be implemented and why.  
Explain the rough plan to implement this feature and what we need.
2. Fork the repository
3. Create a new branch. Remember to give the branch a well describing name
4. Delegate the tasks inbetween the participants
5. Write code. See our style guide to write the most optional code
6. Write tests to see if the features works correctly in most cases
7. Add new libaries to requirements.txt
8. Create a pull request!

## Style guide
### Commits
* Please make sure to add emojies representing the action you made. This will make the commits more visual and faster to scan through.  
```Added``` = ```➕```  
```Fixed``` = ```🔧```  
```Cleaned``` = ```🧹```  
```Removed``` = ```🚮```  
```Updated``` = ```🔁```   
```Bug``` = ```🐛```  
Feel free to add a more fitting emoji if you need to. This is just for inspiration.

### Code
We use the PEP style guide with a slight variation.  
1. Instead of having a line with an end parenthesis, we end the last line with it.  
```python
# Do like this
func(
   arg1 = 'something',
   arg2 = 'i dont know',
   arg3 = 'last argument')
```

2. Use single quotes everytime. This looks much cleaner.

3. Add variable hints to every argument in every function including return type.
```python
# Do it like this
from typing import *
def f(x:Iterable[str], y:int) -> str:
	return x[y*2]
```

4. Good variable naming. Like the previous examples, don't use x and y variables but an actual name.

5. Make a comment describing the function or class. The comment should be placed right beneeth the *def* part. Also include a description of each argument.

6. Write concise comments above complicated code. Use the hashtag comment for this.

