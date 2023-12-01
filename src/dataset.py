import datasets
from collections import defaultdict
import logging
import numpy as np
import copy

def extract_parameters(function: str, language: str) -> list:
    """
    Parses and extracts the parameters from a function
    @param function the function in string representation
    @param language the language the function was coded in
    """
    result = []
    # string before (, and after (, to get parameters within
    header = function.split(")")
    if len(header) <= 1:
        logging.warning("Could not find parameters for this function...\n")
        return result
    # get the parameters
    params = header[0].split("(")[1]
    # isolate each parameter
    params = params.split(",")
    # then, we want to find the commas
    assert(len(params)>= 1)
    for param in params:
        # split by white space
        param_feature_list = param.split(" ")
        # java is our only typed language.
        if language == "java":
            result.append(param_feature_list[1])
        else:
            result.append(param_feature_list[0])
    return result

def get_parameters(examples):
    """
    Batched function to get the parameters of a function.
    @param examples: the examples for our function
    """
    print(type(examples))
    parameters = []
    for example,lang in zip(examples['func_code_string'], examples['language']):
        parameters.append(extract_parameters(example, lang))
    return {"parameters" : parameters}

def criteria(example, len_criteria=5, line_criteria=4):
    """
    The criteria on whether or not an example is good
    @param len_criteria minimum number of words in the documentation
    @param line_criteria exception for code blocks less than this number of lines
    """
    num_lines = example["func_code_tokens"].count("\n")
    doc_len = len(example["func_documentation_tokens"])
    return not (doc_len < len_criteria and num_lines > line_criteria) 

def generic_question(examples, question, answer_fn):
    """
    For a generic question that asks about the overall features of function
    documentation, generate a response and add to the dataset
    @param examples the batch we are mapping over
    @param question the question we want to ask
    @param answer_fn the function that will answer the question give the parameter
    """
    result = dict()
    # copy the entire dataset
    for key, value in examples.items():
        result[key] = copy.deepcopy(value)
    
    # add question column to dataset
    result["question"] = [question]*len(examples)

    answers = []
    for example in examples['func_documentation_string']:
        answers.append(answer_fn(question, example))
    result["answers"] = answers

    return result

def parameter_questions(example, question: str, answer_fn):
    """
    For a generic question that asks about the overall features of function
    documentation, generate a response and add to the dataset
    @param examples the batch we are currently mapping over
    @param question a string formatted question that we can insert the parameter into
    @param answer_fn a function that helps us answer the given question and parameter 
    """



def create_dataset(languages="all", upload= None) -> datasets.Dataset:
    """
    Loads in the dataset and creates the necessary questions
    """
    dataset = datasets.load_dataset("code_search_net", languages)

    # (0) We want to throw away examples where the documentation is bad
    dataset = datasets.filter(criteria)

    # (1) find the parameters for each piece of code and store it
    dataset = datasets.map(get_parameters, batched = True)

    # (2) generate questions 

    
