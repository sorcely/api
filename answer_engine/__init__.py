import sys
sys.path.append('answer_engine/')
sys.path.append('../answer_engine/')

# ML scripts
from pipelines import *

# File management dependcies
import google_storage
import pickle
import json

import os
import re
import random
import time

# Imports for saving results
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='darkgrid')

# Initialize global paths
ANSWER_ENGINE_PATH = 'answer_engine/' if __name__ != '__main__' else '' # Changing the path to files in the answer_engine folder
MODEL_PATH = 'models' if __name__ != '__main__' else '../models/' # Changing the path to files in the model folder

# Just for testing
if __name__ == "__main__":
    answer = pipeline(
        model_names={
            'qa': 'ahotrod/electra_large_discriminator_squad2_512',
            'cs': f'{MODEL_PATH}/GRU-article-crediability.h5'},
        use_custom_pipeline=True,
        from_google_bucket=False)

    start_time = time.time()
    context = 'Denmark is a Nordic country in Northern Europe. Denmark proper, which is the southernmost of the Scandinavian countries, consists of a peninsula, Jutland, and an archipelago of 443 named islands, with the largest being Zealand, Funen and the North Jutlandic Island.'
    question = 'what is denmark'
    a = answer([context], question)
    print(a, '\n', time.time() - start_time)