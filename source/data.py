import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

import logging

logger = logging.getLogger(__name__)

def load_dataset(task="SFT"):
    df = pd.read_csv(os.environ.get("DATA_PATH") + f"input/{task}_training_data.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_data_dict = {"prompt": train_df["instruction"].tolist(), "completion": train_df["output"].tolist()}
    eval_data_dict = {"prompt": val_df["instruction"].tolist(), "completion": val_df["output"].tolist()}

    train_dataset = Dataset.from_dict(train_data_dict)
    eval_dataset = Dataset.from_dict(eval_data_dict)
    logger.info(f"Succesfully loaded train/eval datasets!")
    return train_dataset, eval_dataset