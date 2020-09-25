<p align="center">
  <a href="https://sorcely.tech">
    <img src="https://github.com/sorcely/api-lite/blob/master/.github/logo-lg.png" height="200">  
  </a>
  <br><b>Developing a world without fake-news through state-of-the-art nlp models.</b>
</p>

# Our goal
We're building a fake news detector by using AI and NLP to analyze data uploaded to the internet.  
We aim to integrate this API into every website on the planet - at least every website who wants to be trusted.  
We want to make the fact checking as easy as possible for everyone through AI. 

Made with ❤ from Denmark and the 🌍

# Set up
1. ```git clone https://github.com/sorcely/api-lite.git```
2. ```pip install -r requirements.txt```
3. ```python run.py```
We recommend that you're using python 3.7.x, this is the version we used to create this

## Install on a Linux server
However if you want to install on a Linux server, please run the setup.sh  
```sudo setup.sh```  
This basically does the same, but it assumes that some software hasn't been installed yet. This software include python, pip and git

# Specs
| Engine Type   | Current Time | Preferred Time |
| :---:         | :---:        | :---:          |
| Answer Engine | 2-3s (GPU)   | 1s             |
| Data Engine   | ?            | 0.2s           |
| **In total**  | **2-4s**     | **1.2s**       |

# Goals / roadmap
## Goals for v1.0 release (deadline: september)
* Improve lookup speed  
  ✅Use threading in the webcrawler and translation  
  ✅Use batch prediction (did not improve that much)  
* Improve answer-engine (nlp library)  
  ✅Add Electra QA model  
  ✅Add answer score  
* Improve data-engine (crawler lirary)  
  ❌Add specific websites like snopes.com, factcheck.org  
  ✅Add the News API  

As you can see, we still need to implement the specific websites.  
This along with a better question answering system, is our main goal in October.

## Goals for v2.0 release (deadline: january)
* Improve answer-engine  
  ❌Ensemble a natural questions model with a SQuAD model  
* Improve lookup speed  
  ❌Get even better GPUs  
  ❌Optimize the BERT input pipeline  
* Improve data-engine  
  ❌Find the date, author and publisher    
  ❌Add specific websites like snopes.com, factcheck.org  
* Add a bias-engine  
  ❌Add specific websites like snopes.com, factcheck.org  
 

# Creating a new feature
### Code style
1. Add variable decorations
2. Define what output you should expect
3. Make a comment describing the function or class right. The comment should be placed right beneeth the *def* part
4. Describe the use of the variables inputted into the function or class
5. Write concise comments above complicated code. Use the hashtag for this
6. Use an indention of 4 with spaces

### Commits
* Please make sure to add emojies representing the action you made. This will make the commits more visual and faster  
```Added``` = ```➕```  
```Fixed``` = ```🔧```  
```Cleaned``` = ```🧹```  
```Removed``` = ```🚮```  
```Updated``` = ```🔁```   
Feel free to add a more fitting emoji if you need to. This is just for inspiration.

### Todo list of what to do
* Make sure you've met the requirements from the code style
* Write tests inside the tests folder
* Add dependcies to the requirements.txt
* Add the function into code if not done prior
* Lastly run the tests you created

We're then checking if the output formata and dtypes is correct. And also just checks if it compiles.  
Maybe add more specialized assertions. But we don't believe it's needed yet.

# Contact
If you need help, please open an issue or send an email.

### Emails 
* [Marius, founder](mailto:marius.schlichtkrull@gmail.com)
