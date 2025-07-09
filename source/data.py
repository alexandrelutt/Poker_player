import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

import logging

logger = logging.getLogger(__name__)

def load_dataset(task, tokenizer, config):
    def tokenize_batch(batch):
        return tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=512)

    df = pd.read_csv(os.environ.get("DATA_PATH") + f"input/{task}_training_data.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_data_dict = {"prompt": train_df["instruction"].tolist(), "completion": train_df["output"].tolist(), "ground_truth": train_df["output"].tolist()}
    eval_data_dict = {"prompt": val_df["instruction"].tolist(), "completion": val_df["output"].tolist(), "ground_truth": val_df["output"].tolist()}

    train_dataset = Dataset.from_dict(train_data_dict)
    tokenized_train_dataset = train_dataset.map(tokenize_batch, batched=True)
    eval_dataset = Dataset.from_dict(eval_data_dict)
    tokenized_eval_dataset = eval_dataset.map(tokenize_batch, batched=True)
    logger.info(f"Succesfully loaded train/eval datasets!")
    return tokenized_train_dataset, tokenized_eval_dataset
