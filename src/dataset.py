"""
File for generating our training dataset for QG/QA Code Summarizations
"""
import datasets
from collections import defaultdict
import logging
import copy
import argparse
import re

def extract_parameters(function: str, language: str) -> list:
    """
    Parses and extracts the parameters from a function
    @param function the function in string representation
    @param language the language the function was coded in
    """
    # print(function)
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
    # print(params)
    # print("headed to the list")
    for param in params:
        param = param.strip() # remove whitespace before and after
        # print(param)
        # split by white space
        param_feature_list = param.split(" ")
        # java is our only typed language. 
        if language == "java":
            ans = (param_feature_list[1])
        else:
            ans = (param_feature_list[0])
        if language == "python" and ans == "self":
          continue
        ans = ans.split("=")[0].strip()
        ans = re.sub(r'[^\w]', '', ans)
        result.append(ans)
    return result



def get_parameters(examples):
    """
    Batched function to get the parameters of a function.
    @param examples: the examples for our function
    """
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
    num_lines = example["func_code_tokens"].count(".")
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
        keys = list(examples.data.keys())
        result = defaultdict(lambda: list())
        for key in keys:
          print(key)
          print("length of {} is {}".format(key, len(examples[key])))
        for i in range(len(examples[keys[0]])):
            answer = answer_fn(question, examples["func_documentation_string"][i], examples["func_documentation_tokens"][i])
            if answer != None:
                for key in keys:
                    result[key].append(examples[key][i])
                # then, create the question and the answer
                result["question"].append(question)
                result["answer"].append(answer)


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
        keys = list(examples.data.keys())
        result = defaultdict(lambda: list())

        for i in range(len(examples[keys[0]])):
            # for every parameter in the question, create a new entry
            for parameter in examples["parameters"][i]:
                # attempt to get answer
                answer = answer_fn(parameter, examples["func_documentation_string"][i], examples["func_documentation_tokens"][i])
                
                if answer != None:
                # make a copy of the existing columns in this row
                    for key in keys:
                        if key != "parameters":
                            result[key].append(examples[key][i])
                    # then, create the question and the answer
                    result["question"].append(question.format(parameter))
                    result["answer"].append(answer)
                # don't include if an answer cannot be parsed
        return result

    return inside

            
def generate_answer(question: str, doc_string: str, doc_list: list):
    """
    Generates answer based on the question
    @param question the question in mind, or parameter in question
    @param doc_string the string version of documentation
    @param doc_list tokenized summary. does not contain param or return
    """
    split = doc_string.split(" ")

    if question == "What does this function do?":
        return doc_list
    if question == "What does this function return?":
        idxs= [i for i, item in enumerate(split) if re.search("\W*.return[\S]?\W*", item)]
        if len(idxs) != 1:
            return None
        # get the subset. we assume that this is at the end of the document.
        subset = split[idxs[0]+1:]
        if len(subset) == 0:
          return None
        return subset
    else:
        idxs = [i for i, item in enumerate(split) if re.search("\W*.param[\S]?\W*", item)]
        for idx in idxs:
            pattern = "{}.?".format(question)
            if (idx < len(split) - 1) and re.match(pattern, split[idx + 1], re.IGNORECASE):
                result = []
                curr = idx + 2
                while curr < len(split):
                    result.append(split[curr])
                    if re.match("(\\w+)(?=[.])]", split[curr]) != None:
                        return result
                    curr += 1
        return None


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