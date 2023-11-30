import datasets
from collections import defaultdict
import logging
import numpy as np


def extract_parameters(function: str, language: str) -> list:
    """
    We'll take in the whole string, and perform our own tokenization
    to parse them
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
    print(type(examples))
    parameters = []
    for example,lang in zip(examples['func_code_string'], examples['language']):
        parameters.append(extract_parameters(example, lang))
    return {"parameters" : parameters}



    # we first isolate the parameters by finding the first occurence of 
    # "(" and ")". we then subset 
    # from there, we find every occurence of "," and split on it 
    # then if the language is not type checked we take the first occurence
    # of each split. otherwise we take the second occurence
    # we will then use regex to return anything that isn't alphanumeric
    # and then return the parameters as a list

def extract_answer(documentation_string: str) -> str:
    line_ends = [".", "\n", ]


def convert_to_question(examples : dict, question : str, answer_parser_func, parameters=False):
    # if the question is based on parameters, we will want to augment the dataset
    # so we'll do some fancy stuff.
    # otherwise, simply append the answer
    questions = [question]*len(examples)
    answers = []
    for example in examples["func_documentation_string"]:
        answer = answer_parser_func(example)
        answers.append(answer)
    examples["question"] = questions
    examples["answers"] = answers

    return examples






def load_dataset(languages="all"):
    """
    loads the dataset.
    """
    dataset = datasets.load_dataset("code_search_net", languages)
