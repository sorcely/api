import torch
import transformers
from transformers import pipeline as huggingface_pipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Imports the answering modules for the sorcely pipeline
from answering import question_answering
from answering import *
import crediability
from crediability import crediability_scoring

# Returns an initialized pipeline ready for inference
# This both include the sorcely pipeline and the huggingface extended pipeline
def pipeline(model_names:dict, use_custom_pipeline:bool=False, from_google_bucket:bool=False):
    # model_name: list of names of the models we're using
    # use_custom_pipeline: Whether to use Huggingface's pipeline or our own pipeline
    # from_google_bucket: Where to download the model files from

    # Model names
    qa_model_name = model_names['qa']
    cs_model_name = model_names['cs']

    # Downloads the question answering model
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

    cs_model = crediability.model(cs_model_name, crediability.config)
    cs_tokenizer = crediability.tokenizer(crediability.config)

    # Defines what hardware to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialized the chosen pipeline
    if not use_custom_pipeline:
        pipeline_fn = huggingface_pipeline_ext(
            models={'qa': qa_model, 'cs': cs_model},
            tokenizers={'qa': qa_tokenizer, 'cs': cs_tokenizer},
            device=device)
    else:
        pipeline_fn = sorcely_pipeline(
            models={'qa': qa_model, 'cs': cs_model},
            tokenizers={'qa': qa_tokenizer, 'cs': cs_tokenizer})

    return pipeline_fn

# Our custom written pipeline for both article scoring and question answering
class sorcely_pipeline:

    def __init__(self, models:list, tokenizers:list):
        # model: [The preloaded qa model, preloaded crediability model]
        # tokenizer: [The preloaded qa tokenizer, preloaded crediability tokenizer]

        self.models = models
        self.tokenizers = tokenizers

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        qa_model = self.models['qa']
        qa_model.to(device)
        qa_tokenizer = self.tokenizers['qa']

        cs_model = self.models['cs']
        cs_tokenizer = self.tokenizers['cs']

        self.question_answering = question_answering(qa_model, qa_tokenizer)
        self.crediability_scoring = crediability_scoring(cs_model, cs_tokenizer)

    def __call__(self, contexts, question):
        # contexts: A list of contexts we want to extract an answer from
        # question: The question we want answered

        answers = self.question_answering(contexts, question, max_len=256)
        crediabilities = self.crediability_scoring(contexts)

        # Concatenates the answers with the crediabilities
        for i,_ in enumerate(answers):
            answers[i] = list(answers[i])
            answers[i].append(crediabilities[i])

        return answers

# Extending the Huggingface pipeline by adding batches
class huggingface_pipeline_ext:

    def __init__(self, models:list, tokenizers:list, device):
        # models: [The preloaded QA model, pre
        # tokenizers: The preloaded tokenizer
        # device: The device we want our model to run on

        qa_model = models['qa']
        qa_model.to(device)
        qa_tokenizer = tokenizers['qa']

        cs_model = models['cs']
        cs_tokenizer = tokenizers['cs']

        # Initializes the question answering pipeline
        self.question_answering = huggingface_pipeline(
            'question-answering',
            model=qa_model,
            tokenizer=qa_tokenizer,
            device=int(torch.cuda.is_available()) - 1)

        # Initializes the crediability scoring pipeline
        try:
            self.crediability_scoring = crediability_scoring(
                model=cs_model,
                tokenizer=cs_tokenizer)
            self.crediability_scoring_implemented = True
        except:
            self.crediability_scoring_implemented = False

    def __call__(self, contexts:list, question:str):
        # contexts: A list of contexts we want to extract an answer from
        # question: The question we want answered

        answers = []
        for context in contexts:
            # Gets an answer using the Huggingface QA pipeline
            answer = self.question_answering(context=context, question=question)
            answers.append([answer['answer'], answer['score']])

        # Gets the article crediability using a custom written pipeline
        # This uses batch prediction unlike the huggingface pipeline, 
        # thus, we can just do it once without a for-loop
        if self.crediability_scoring_implemented:
            crediabilities = self.crediability_scoring(contexts)
        else:
            crediabilities = [1.0] * len(contexts)

        # Concatenates the answers with the crediabilities
        for i,_ in enumerate(answers):
            answers[i].append(crediabilities[i])

        return answers
