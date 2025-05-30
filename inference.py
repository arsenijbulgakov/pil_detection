import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import json
import argparse
from itertools import chain
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset
import numpy as np


from omegaconf import DictConfig, OmegaConf
import hydra



def tokenize(example, tokenizer, max_length):
    text = []
    token_map = []
    
    idx = 0
    
    for t, ws in zip(example["tokens"], example["trailing_whitespace"]):
        
        text.append(t)
        token_map.extend([idx]*len(t))
        if ws:
            text.append(" ")
            token_map.append(-1)
            
        idx += 1
        
        
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, truncation=True, max_length=max_length)
    
        
    return {
        **tokenized,
        "token_map": token_map,
    }





@hydra.main(version_base=None, config_path="./configs", config_name="infer_config")
def main(cfg: DictConfig):


    #parser = argparse.ArgumentParser()
    #parser.add_argument('--model-path', required=False, default='./deberta3base_512', type=str)
    #parser.add_argument('--max-length', required=False, default=2048, type=int)
    #parser.add_argument('--test-dataset-path', required=False, default="./test.json", type=str)
    #parser.add_argument('--predictions-path', required=False, default="submission.csv", type=str)

    #args = parser.parse_args()


    data = json.load(open(cfg["test_dataset_path"]))
    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [x["document"] for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    })
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "max_length": cfg["max_length"]}, num_proc=2)


    model = AutoModelForTokenClassification.from_pretrained(cfg["model_path"])
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
    trainer_args = TrainingArguments(
        ".", 
        per_device_eval_batch_size=1, 
        report_to="none",
    )
    trainer = Trainer(
        model=model, 
        args=trainer_args, 
        data_collator=collator, 
        tokenizer=tokenizer,
    )


    predictions = trainer.predict(ds).predictions
    pred_softmax = np.exp(predictions) / np.sum(np.exp(predictions), axis = 2).reshape(predictions.shape[0],predictions.shape[1],1)
    config = json.load(open(Path(cfg["model_path"]) / "config.json"))
    id2label = config["id2label"]
    preds = predictions.argmax(-1)
    preds_without_O = pred_softmax[:,:,:12].argmax(-1)
    O_preds = pred_softmax[:,:,12]
    threshold = 0.9
    preds_final = np.where(O_preds < threshold, preds_without_O , preds)



    triplets = []
    document, token, label, token_str = [], [], [], []
    for p, token_map, offsets, tokens, doc in zip(preds_final, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]):

        for token_pred, (start_idx, end_idx) in zip(p, offsets):
            label_pred = id2label[str(token_pred)]

            if start_idx + end_idx == 0: continue

            if token_map[start_idx] == -1:
                start_idx += 1

            # ignore "\n\n"
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1

            if start_idx >= len(token_map): break

            token_id = token_map[start_idx]

            # ignore "O" predictions and whitespace preds
            if label_pred != "O" and token_id != -1:
                triplet = (label_pred, token_id, tokens[token_id])

                if triplet not in triplets:
                    document.append(doc)
                    token.append(token_id)
                    label.append(label_pred)
                    token_str.append(tokens[token_id])
                    triplets.append(triplet)



    df = pd.DataFrame({
        "document": document,
        "token": token,
        "label": label,
        "token_str": token_str
    })
    df["row_id"] = list(range(len(df)))
    df[["row_id", "document", "token", "label"]].to_csv(cfg["predictions_path"], index=False)




if __name__ == "__main__":
    main()