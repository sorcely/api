import torch
import transformers
from transformers import pipeline as huggingface_pipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Imports the answering modules for the sorcely pipeline
from answering import question_answering
from answering import *

from typing import Callable

# Returns an initialized pipeline ready for inference
# This both include the sorcely pipeline and the huggingface extended pipeline
def pipeline(model_names:dict, use_custom_pipeline:bool = False, from_google_bucket:bool = False) -> Callable:
    '''
    Creates and returns a function to run the Question Answering AI

    ### Args ###
    model_names (:obj: `dict`)
        * A dictionary containing the names of the model we want to use
        * the qa key is specifying the name of the question answering model
    use_custom_pipeline (:obj: `str`)
        * Defines what pipeline type we want to use. Either a "custom" written pipeline or the official Huggingface Transformers pipeline.
    from_google_bucket (:obj: `str`)
        * Tells us if we should download the model from a google bucket
    '''

    # Model names
    qa_model_name = model_names['qa']

    # Downloads the question answering model
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

    # Defines what hardware to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialized the chosen pipeline
    if not use_custom_pipeline:
        pipeline_fn = huggingface_pipeline_ext(
            models = {'qa': qa_model},
            tokenizers = {'qa': qa_tokenizer},
            device = device)
    else:
        pipeline_fn = sorcely_pipeline(
            models = {'qa': qa_model},
            tokenizers = {'qa': qa_tokenizer})
    return pipeline_fn

# Our custom written pipeline for both article scoring and question answering
class sorcely_pipeline:

    def __init__(self, models:dict, tokenizers:dict):
        '''
        Creates and returns a function to run the Question Answering AI

        ### Args ###
        models (:obj: `dict`)
            * Provides the function with a preloaded PretrainedModel
            * Currently we only require qa, and all other models is just wasting our memory
        tokenizer (:obj: `dict`)
            * Provides the function with a preloaded tokenizer
            * Currently we only require qa, and all other tokenizers is just wasting our memory
        '''

        self.models = models
        self.tokenizers = tokenizers

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        qa_model = self.models['qa']
        qa_model.to(device)
        qa_tokenizer = self.tokenizers['qa']

        self.question_answering = question_answering(qa_model, qa_tokenizer)

    def __call__(self, contexts:list, question:str) -> list:
        '''
        Creates and returns a function to run the Question Answering AI

        ### Args ###
        contexts (:obj: `list`)
            * A list of all the scraped data from the webpages
            * It's expected to be translated into english
        question (:obj: `str`)
            * The english question we want answered
        '''

        answers = self.question_answering(contexts, question, max_len = 256)

        # Concatenates the answers with the crediabilities
        for i,_ in enumerate(answers):
            answers[i] = list(answers[i])

        return answers

# Extending the Huggingface pipeline by adding batches
class huggingface_pipeline_ext:

    def __init__(self, models:dict, tokenizers:dict, device:torch.device):
        '''
        Creates and returns a function to run the Question Answering AI

        ### Args ###
        models (:obj: `dict`)
            * Provides the function with a preloaded PretrainedModel
            * Currently we only require qa, and all other models is just wasting our memory
        tokenizer (:obj: `dict`)
            * Provides the function with a preloaded tokenizer
            * Currently we only require qa, and all other tokenizers is just wasting our memory
        device (:obj: `torch.device`)
            * The device the model will be running on
        '''

        qa_model = models['qa']
        qa_model.to(device)
        qa_tokenizer = tokenizers['qa']

        # Initializes the question answering pipeline
        self.question_answering = huggingface_pipeline(
            'question-answering',
            model=qa_model,
            tokenizer=qa_tokenizer,
            device=int(torch.cuda.is_available()) - 1)

    def __call__(self, contexts:list, question:str) -> list:
        '''
        Creates and returns a function to run the Question Answering AI

        ### Args ###
        contexts (:obj: `list`)
            * A list of all the scraped data from the webpages
            * It's expected to be translated into english
        question (:obj: `str`)
            * The english question we want answered
        '''

        answers = []
        for context in contexts:
            # Gets an answer using the Huggingface QA pipeline
            answer = self.question_answering(context=context, question=question)
            answers.append([answer['answer'], answer['score']])

        return answers
