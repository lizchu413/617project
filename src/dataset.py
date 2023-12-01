"""
File for generating our training dataset for QG/QA Code Summarizations
"""
import datasets
from collections import defaultdict
import logging
import copy
import argparse

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

def generic_question(question: str, answer_fn) -> dict:
    """
    For a generic question that asks about the overall features of function
    documentation, generate a response and add to the dataset
    This is a wrapper function for dataset.map
    @param examples the batch we are mapping over
    @param question the question we want to ask
    @param answer_fn the function that will answer the question give the parameter
    """
    def inside(examples):
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
    return inside

def parameter_question(question: str, answer_fn) -> dict:
    """
    For a generic question that asks about the overall features of function
    documentation, generate a response and add to the dataset.
    Notice how we take in an example, since we can no longer batch over these
    This is a wrapper function for dataset.map
    @param example the item we are taking in 
    @param question a string formatted question that we can insert the parameter into
    @param answer_fn a function that helps us answer the given question and parameter 
    """
    def inside(examples):
        keys = examples.keys()
        result = defaultdict(lambda: list())
        for i in range(len(keys[0])):
            # for every parameter in the question, create a new entry
            for parameter in examples["parameters"][i]:
                # make a copy of the existing columns in this row
                for key in keys:
                    if key != "parameters":
                        result[key] = examples[key][i]
                # then, create the question and the answer
                result["question"].append(question.format(parameter))
                result["answer"] = answer_fn(result["question"][-1], result["func_documentation_string"][-1])
        return result

    return inside

            
def generate_answer(question: str, context: str):
    return "dummy"


def create_dataset(languages="all", upload=None) -> datasets.Dataset:
    """
    Loads in the dataset and creates the necessary questions
    """
    logging.info("Loading in dataset...\n")
    dataset = datasets.load_dataset("code_search_net", languages)
    logging.info(f"Finished loading in dataset. This dataset contains {len(dataset)} entries\n")

    # (0) We want to throw away examples where the documentation is bad
    dataset = datasets.filter(criteria)
    logging.info(f"Finished filtering dataset for bad examples. This dataset now contains {len(dataset)} entries\n")

    logging.info("Now parsing out the parameters of each function...\n")
    # (1) find the parameters for each piece of code and store it
    dataset = datasets.map(get_parameters, batched = True)
    logging.info("Finished parsing.")

    # (2) generate questions and answers
    logging.info("Generating dataset for first question....\n")
    do_dataset = dataset.map(generic_question("What does this function do?", generate_answer))
    logging.info("Generating dataset for second question....\n")
    param_dataset = dataset.map(parameter_question("What is {}?", generate_answer)) 
    logging.info("Generating dataset for third question....\n")
    return_dataset = dataset.map(generic_question("What does this function return?", generate_answer))
    
    # (3) concatenate datasets
    logging.info("Concatenating datasets.\n")
    complete_dataset = datasets.concatenate_datasets([do_dataset, param_dataset, return_dataset])

    # (4) upload to hub
    if upload != None:
        logging.info("Uploading to the hub.\n")
        complete_dataset.push_to_hub(upload)
    
    logging.info("All complete.\n")
    return complete_dataset

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "%(levelname)s | %(asctime)s | %(message)s")
    # parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', help="The HuggingFace repository to upload to")
    parser.add_argument('--lang', choices=['python', 'php','go', 'javascript', 'java', 'ruby', 'all'], default= 'all')
    args = parser.parse_args()

    create_dataset(args.lang, args.upload)