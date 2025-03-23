import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import json
import argparse
from itertools import chain
from functools import partial

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
import evaluate
from datasets import Dataset, features
import numpy as np


from omegaconf import DictConfig, OmegaConf
import hydra

import mlflow


def tokenize(example, tokenizer, label2id, max_length):

    # rebuild text from tokens
    text = []
    labels = []

    for t, l, ws in zip(
        example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    ):
        text.append(t)
        labels.extend([l] * len(t))

        if ws:
            text.append(" ")
            labels.append("O")

    # actual tokenization
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length, truncation=True)

    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {**tokenized, "labels": token_labels, "length": length}




from seqeval.metrics import recall_score, precision_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

def compute_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5*5) * recall * precision / (5*5*precision + recall)
    
    results = {
        'recall': recall,
        'precision': precision,
        'f1': f1_score
    }
    return results




@hydra.main(version_base=None, config_path="./configs", config_name="train_config")
def main(cfg: DictConfig):

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--model-path-hf', required=False, default='microsoft/deberta-v3-base', type=str)
    #parser.add_argument('--max-length', required=False, default=512, type=int)
    #parser.add_argument('--output-dir', required=False, default="test_trainer_log", type=str)
    #parser.add_argument('--train-dataset-path', required=False, default="train.json", type=str)
    #parser.add_argument('--output-model-path', required=False, default="deberta3base_512", type=str)
    #args = parser.parse_args()

    mlflow.set_experiment(cfg["mlflow"]["exp_name"])

    data = json.load(open(cfg["data"]["train_dataset_path"]))

    all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
    label2id = {l: i for i,l in enumerate(all_labels)}
    id2label = {v:k for k,v in label2id.items()}

    target = [
        'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 
        'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 
        'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL'
    ]


    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_path_hf"])

    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [str(x["document"]) for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data],
    })
    ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": cfg["max_length"]}, num_proc=3)




    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model"]["model_path_hf"],
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)


    trainer_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        **cfg["training"]
    )

    trainer = Trainer(
        model=model, 
        args=trainer_args, 
        train_dataset=ds,
        eval_dataset=ds,
        data_collator=collator, 
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, all_labels=all_labels),
    )



    with mlflow.start_run(run_name = cfg["mlflow"]["run_name"]) as run:
        #mlflow.log_metric("score1", 100)
        #mlflow.log_metric("score2", 200)
        #mlflow.log_metric("score3", 300)
        
        trainer.train()


    trainer.save_model(cfg["model"]["output_model_path"])
    tokenizer.save_pretrained(cfg["model"]["output_model_path"])


    




if __name__ == "__main__":
    main()