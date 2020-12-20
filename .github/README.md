<p align="center">
  <a href="https://sorcely.tech">
    <img src="https://github.com/sorcely/api-lite/blob/master/.github/logo-lg.png" height="200">  
  </a>
  <br><b>Developing a world without fake-news through state-of-the-art nlp models.</b>
</p>

# Our goal
We're building a fake news detector by using AI and NLP to analyse data uploaded to the internet.  
We aim to integrate this API into every website on the planet - at least every website who wants to be trusted.  
We want to make the fact checking as easy as possible for everyone through AI. 

# Set up
0. Fork the repository
1. ```git clone https://github.com/$username/api-lite.git```
2. ```pip install -r requirements.txt```
3. ```python run.py```
We recommend that you're using python 3.7.x.

### Install on a Linux server
However if you want to install on a Linux server, please run the setup.sh  
```sudo setup.sh```  
This basically does the same, but it assumes that some software hasn't been installed yet. This software that will be installed is: ```python```, ```pip``` & ```git```

# Speed
| Engine Type   | Current Time | Preferred Time |
| :---:         | :---:        | :---:          |
| Answer Engine | 2-3s (GPU)   | 1s             |
| Data Engine   | 0.5s         | 0.2s           |
| **In total**  | **2-4s**     | **1.2s**       |

# Goals / roadmap
## Goals for v2.0 release (deadline: february)
* Improve answer-engine  
  ❌Ensemble a natural questions model with a SQuAD model  
* Improve lookup speed  
  ❌Get even better GPUs & TPUs  
  ❌Optimize the BERT input pipeline  
* Cheaper to run  
  ❌Write GPU sharing software  
* Improve data-engine  
  ❌Render websites' JavaScript  
  ❌Find the date, author and publisher  
  ❌Add specific websites like snopes.com, factcheck.org  
* Add a bias-engine  
  ❌Create bias dataset  
  ❌Train a bias detection model  
  ❌Create bias pipeline  

# Our lovely community
## Shoutout to all stargazers and contributors!
[![Stargazers repo roster for @sorcely/api](https://reporoster.com/stars/sorcely/api)](https://github.com/sorcely/api/stargazers)
[![Forkers repo roster for @sorcely/api](https://reporoster.com/forks/sorcely/api)](https://github.com/sorcely/api/network/members)

# Contact
If you need help, please open an issue or send an email.

### Emails 
* [Marius J. Schlichtkrull, founder](mailto:marius.schlichtkrull@gmail.com)
