from datasets import Dataset
from ptdata import CodeDataset
import torch.utils.data as torchdata

FILEPATH = "../code-docstring-corpus-V2/repo_split/repo_split.parallel_{}.{}"


def fetch_data(split="train") -> list:
    """
    Fetches data from local path and returns as a tuple of (description, declaration, body)
    """
    result = []
    components = ["desc", "decl", "bodies"]
    for component in components:
        with open(
            FILEPATH.format(component, split),
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:
            result.append([line[1:-2] for line in file])

    return result


def get_length(examples: dict) -> dict:
    """
    Given a batch of examples, returns the number of lines in the code body based on new line character count.
    """
    lengths = []
    for codeblock in examples["body"]:
        words = codeblock.split(" ")
        count = words.count("DCNL")
        lengths.append(count + 1)
    return {"length": lengths}


def build_hf_dataset(
    descriptions: list, declarations: list, bodies: list, upload=None
) -> Dataset:
    """
    Converts lists into a huggingface dataset, and uploads if needed
    """
    dataset_dict = dict()
    dataset_dict["body"] = list(bodies)
    dataset_dict["desc"] = list(descriptions)
    dataset_dict["decl"] = list(declarations)

    result = Dataset.from_dict(dataset_dict)
    result = result.map(get_length, batched=True)

    if upload != None:
        result.push_to_hub(upload)
    return result


def build_pt_dataset(
    descriptions: list, declarations: list, bodies: list, upload=None
) -> torchdata.Dataset:
    """
    returns a PyTorch dataset based on the lists given.
    """
    return CodeDataset(bodies, descriptions, declarations)


if __name__ == "__main__":
    train_desc, train_decl, train_bodies = fetch_data("train")
    dataset = build_hf_dataset(train_desc, train_decl, train_bodies)
    print(dataset[:5])
