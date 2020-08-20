<p align="center">
  <a href="https://sorcely.tech">
    <img src="https://github.com/mariusjohan/Sorcely/blob/master/.github/logo-lg.png" height="115">  
  </a>
  <br><b>Developing a world without fake-news through state-of-the-art nlp models.</b>
</p>

# 📜 About
We've built an fact checker using a variety of NLP models. We believe to have an as big impact as possible, we want to make every website able to use this software, even for free. However it's best to use GPUs for this softwase. We're using a lot of ML and thus a lot of compute. You can either use your own GPU server, or you can choose to buy a subscription to our servers.  
We hope you'd implement this into your own website, and we'd love to have you contributing to our project.  
Together we can chance the world. 

- Made with love from Denmark and the rest of the world.

# 📥 Installation
1. ```git clone https://github.com/sorcely/api.git```
2. ```pip install -r requirements.txt```
3. ```python run.py```

However if you want to install on a Linux server, please run the setup.sh  
```sudo setup.sh```  
This basically does the same, but it assumes Python, git and so on hasn't been installed yet.

# ⏱ Time per lookup
| Engine Type   | Current Time | Preferred Time |
| :---:         | :---:        | :---:          |
| Answer Engine | 2-3s (GPU)   | 1s             |
| Data Engine   | ?            | 0.2s           |
| **In total**  | **20-30s**   | **1.2s**       |

# 🎯 Goals for v1.0 release (before september)
* Improve lookup speed  
  ❌Use GPUs for the AI (speed increase up to 120x)  
  ✅Use threading in the webcrawler and translation  
  ✅Use batch prediction (did not improve that much)  
* Improve answer-engine (nlp library)  
  ✅Add Electra QA model  
  ✅Add answer score  
* Improve data-engine (crawler lirary)  
  ❌Add specific websites like snopes.com, factcheck.org  
  ✅Add the News API  

# 🔧 Creating a new feature
**Requirements when creating a new feature**
1. Write clean and well commented code
2. Write tests for the newly created code
3. Implement the feature into the API
4. Add new dependcies to requirements.txt
5. Test the function in tests/test_$MODULE.py  
Where module could be *answer-engine*, *data-engine*, *run* or *server*

**Pushing the code**
1. Write a description of the feature you'll like to have
2. Write pros and cons
3. Document the results on the testset

# 🧪 Testing
You can either choose to run all tests at once, a smaller test on just the run function or run a specific tests for a module like answer-engine or data-engine.

#### Tests you can run
* tests/test.py
* tests/test_run.py
* tests/test_server.py
* tests/test_answer_engine.py
* tests/test_data_engine.py

We're then checking if the output formata and dtypes is correct. And also just checks if it compiles.  
Maybe add more specialized assertions. But we don't believe it's needed yet.

# 📧 Contact
If you need help, please open an issue or send an email.

### Emails 
* [Marius, founder](mailto:marius.schlichtkrull@gmail.com)
