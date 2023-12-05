"""
File for generating our training dataset for QA Code Summarizations
"""
import datasets
import logging
import argparse
import re

def extract_parameters(function: str, language: str) -> list:
    """
    Parses and extracts the parameters from a function
    @param function the function in string representation
    @param language the language the function was coded in
    @return list of parameters for one function
    """
    result = []
    # string before ), and after (, to get parameters within
    header = function.split(")")
    if len(header) <= 1:
        logging.warning(f"Could not find parameters for this {language} function...\n")
        return result
    # get the parameters
    params = header[0].split("(")[1]
    # isolate each parameter
    params = params.replace(", ", ",")
    params = params.split(",")
    # if no parameters
    if(params == [""]):
        return result
    for param in params:
        param = param.strip() # remove whitespace before and after
        # split by white space
        param_feature_list = param.split(" ")
        # java is our only typed language. 
        if language == "java":
            try:
                ans = (param_feature_list[1])
            except:
                return [] # give up
        else:
            ans = (param_feature_list[0])
        # language-specific exception
        if language == "python" and ans == "self":
          continue
        # remove default values
        ans = ans.split("=")[0].strip()
        # remove special markers (&, *, etc.)
        ans = re.sub(r'[^\w]', '', ans)
        result.append(ans)
    return result

def check_ruby_parameters(function: list) -> bool:
    """
    Because Ruby is a stupid language with stupidn rules
    You don't need to include the parentheses if you don't 
    have any arguments, so we'll do it here.
    @param tokenized function
    @return whether function has no parameters
    """
    return function[2] != "("


def get_parameters(examples) -> dict:
    """
    Batched function to get the parameters of a function.
    @param examples: the examples for our function
    @return the parameters of each example
    """
    parameters = []
    for i, (example,lang) in enumerate(zip(examples['func_code_string'], examples['language'])):
        if(lang == "ruby") and check_ruby_parameters(examples['func_code_tokens'][i]):
            parameters.append([])
        else:
            parameters.append(extract_parameters(example, lang))
    return {"parameters" : parameters}

def criteria(example, len_criteria=5, line_criteria=4)-> bool:
    """
    The criteria on whether or not an example is good
    @param len_criteria minimum number of words in the documentation
    @param line_criteria exception for code blocks less than this number of lines
    @return whether the example meets the criteria
    """
    num_lines = max(example["func_code_tokens"].count("."), example["func_code_tokens"].count("\n"))
    doc_len = len(example["func_documentation_tokens"])
    return not (doc_len < len_criteria and num_lines > line_criteria) 

def generic_question(question: str, answer_fn):
    """
    For a generic question that asks about the overall features of function
    documentation, generate a response and add to the dataset
    This is a wrapper function for dataset.map
    @param examples the batch we are mapping over
    @param question the question we want to ask
    @param answer_fn the function that will answer the question give the parameter
    @answer new rows with question and answer
    """
    def inside(examples) -> dict:
        keys = list(examples.data.keys())
        res_keys = keys + ["question", "answer"]
        result = initialize_batch(res_keys)
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


def parameter_question(question: str, answer_fn):
    """
    For a generic question that asks about the overall features of function
    documentation, generate a response and add to the dataset.
    Notice how we take in an example, since we can no longer batch over these
    This is a wrapper function for dataset.map
    @param example the item we are taking in 
    @param question a string formatted question that we can insert the parameter into
    @param answer_fn a function that helps us answer the given question and parameter 
    @return new rows with question and answer
    """
    def inside(examples) -> dict:
        keys = list(examples.data.keys())
        res_keys = keys + ["question", "answer"]
        result = initialize_batch(res_keys)
        for i in range(len(examples[keys[0]])):
            # for every parameter in the question, create a new entry
            for parameter in examples["parameters"][i]:
                # attempt to get answer
                answer = answer_fn(parameter, examples["func_documentation_string"][i], examples["func_documentation_tokens"][i])      
                if answer != None:
                  for key in keys:
                    result[key].append(examples[key][i])
                  # create the question and the answer
                  result["question"].append(question.format(parameter))
                  result["answer"].append(answer)
                # don't include if an answer cannot be parsed
        return result

    return inside

            
def generate_answer(question: str, doc_string: str, doc_list: list)->list:
    """
    Generates answer based on the question
    @param question the question in mind, or parameter in question
    @param doc_string the string version of documentation
    @param doc_list tokenized summary. does not contain param or return
    @return tokenized answer to the question
    """

    # get rid of new lines
    split = doc_string.replace("\n", " ").split(" ")
    # CASE 1
    if question == "What does this function do?":
        # may want to change this because the dataset SUCKS.
        return doc_list
    
    # CASE 2
    if question == "What does this function return?":
        idxs= [i for i, item in enumerate(split) if re.search("\W*.return[\S]?\W*", item)]
        if len(idxs) != 1:
            return None
        # get the subset. we assume that this is at the end of the document.
        subset = split[idxs[0]+1:]
        if len(subset) == 0:
          return None
        return subset
    
    # CASE 3
    else:
        # find all special markers of param
        idxs = [i for i, item in enumerate(split) if re.search("\W*.param[\S]?\W*", item)]

        # search to special markers to find the one equal to ours
        for idx in idxs:
            pattern = "{}.?".format(question)
            if (idx < len(split) - 1) and re.match(pattern, split[idx + 1], re.IGNORECASE):
                result = []
                curr = idx + 2
                # read tokens after special marker until EOS or next special marker
                while curr < len(split):
                    if re.match("^@", split[curr]) or re.match("^:", split[curr]):
                      return result
                    result.append(split[curr])
                    if split[curr].endswith(".") or split[curr].endswith("\n"):
                        return result
                    curr += 1
                if len(result) > 0:
                  return result
        return None
    
def initialize_batch(keys)-> dict:
    """
    Initializes an empty batch
    @param keys the columns of the batch
    @return the resulting empty batch
    """
    result = dict()
    for key in keys:
        result[key] = []
    return result

def create_dataset(languages="all", upload=None) -> datasets.Dataset:
    """
    Loads in the dataset and creates the necessary questions
    @param languages language to upload from the huggingface dataset
    @param upload url of upload repository or None
    @return the dataset
    """
    logging.info("Loading in dataset...\n")
    dataset = datasets.load_dataset("code_search_net", languages)
    logging.info(f"Finished loading in dataset. This dataset contains {dataset.shape} entries\n")

    # (0) We want to throw away examples where the documentation is bad
    dataset = dataset.filter(criteria)
    logging.info(f"Finished filtering dataset for bad examples. This dataset now contains {dataset.shape} entries\n")

    logging.info("Now parsing out the parameters of each function...\n")
    # (1) find the parameters for each piece of code and store it
    dataset = dataset.map(get_parameters, batched = True)
    logging.info("Finished parsing.")

    # (2) generate questions and answers
    logging.info("Generating dataset for first question....\n")
    do_dataset = dataset.map(generic_question("What does this function do?", generate_answer), batched = True, remove_columns = dataset['train'].column_names)
    logging.info("Generating dataset for second question....\n")
    param_dataset = dataset.map(parameter_question("What is {}?", generate_answer), batched = True, remove_columns = dataset['train'].column_names) 
    logging.info("Generating dataset for third question....\n")
    return_dataset = dataset.map(generic_question("What does this function return?", generate_answer), batched = True, remove_columns = dataset['train'].column_names)
    
    # (3) concatenate datasets
    logging.info("Concatenating datasets.\n")
    dataset_list = [do_dataset, param_dataset, return_dataset]
    complete_dataset = datasets.DatasetDict()
    for key in ["train", "test", "validation"]:
        complete_dataset[key] = datasets.concatenate_datasets([ddd[key] for ddd in dataset_list])
    # complete_dataset = datasets.concatenate_datasets([do_dataset, param_dataset, return_dataset])

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