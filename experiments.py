from codet5_to_qa_new import train, format_and_tokenize
from datasets import load_dataset
from transformers import TrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

CODET5 = "Salesforce/codet5p-220m"
BERT = "microsoft/codebert-base"
BATCH_SIZE = 16
EPOCHS = 2
SAVE_EVERY = 1
LR = 1e-4
WORKERS = 2
MAX_INPUT_LENGTH = 120
SEED = 617


def load_experiment(model_out: str, loss_out:str, lr=5e-5, weight_decay=0):
    result = dict()
    result["model_out"] = model_out
    result["loss_out"] = loss_out
    result["lr"] = lr
    result["wd"] = weight_decay
    return result

def experiments(model_name, train_set, val_set, test_set, parameter_sets: list) -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for parameters in parameter_sets:
        args = TrainingArguments(
            output_dir="codet5-to-qa", 
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=parameters["lr"],
            weight_decay = parameters["wd"],
            per_device_train_batch_size=32,
            per_device_eval_batch_size=2,
            num_train_epochs=EPOCHS,
            load_best_model_at_end=True,
            # fp16 = True
        )
        torch.cuda.empty_cache()
        train(model, tokenizer, train_set, val_set, test_set, args, parameters["model_out"], parameters["loss_out"])

if __name__ == "__main__":

    # load in dataset
    _data = load_dataset("aalexchengg/codesearchnet_qa", streaming = True)
    train_set, val_set, test_set = format_and_tokenize(_data, num_train = 4000, num_val=100, num_test=100)

    # hyperparams to tune
    learning_rates = [5e-4,5e-5,5e-6]
    weight_decay = [0.001, 0.0001]

    # code t5 experiments

    # generate experiments
    t5_lr_experiments = [load_experiment("liz_codet5_lr{}.model".format(str(lr).replace(".", "")), "liz_codet5_lr{}.txt".format(str(lr).replace(".", "")), lr, 0) for lr in learning_rates]
    t5_wd_experiments = [load_experiment("liz_codet5_wd{}.model".format(str(wd).replace(".", "")), "liz_codet5_wd{}.txt".format(str(wd).replace(".", "")), 5e-5, wd) for wd in weight_decay]
    # run experiments
    experiments(CODET5, train_set, val_set, test_set, t5_lr_experiments)
    experiments(CODET5, train_set, val_set, test_set, t5_wd_experiments)
    # codebert experiments

    bert_lr_experiments = [load_experiment("liz_bert_lr{}.model".format(str(lr).replace(".", "")), "liz_bert_lr{}.txt".format(str(lr).replace(".", "")), lr, 0) for lr in learning_rates]
    bert_wd_experiments = [load_experiment("liz_bert_wd{}.model".format(str(wd).replace(".", "")), "liz_bert_wd{}.txt".format(str(wd).replace(".", "")), 5e-5, wd) for wd in weight_decay]

    # run experiments
    experiments(BERT, train_set, val_set, test_set, t5_lr_experiments)
    experiments(BERT, train_set, val_set, test_set, t5_wd_experiments)

