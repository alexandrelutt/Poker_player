import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

import logging

def get_last_checkpoint(folder):
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name="SmolLM2-135M-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        os.environ.get("DATA_PATH") + f"models/{model_name}",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("DATA_PATH") + f"models/{model_name}")
    logger.info(f"Succesfully loaded {model_name}!")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.bos_token   
    return model, tokenizer

def load_model_checkpoint_and_tokenizer(model_name="SmolLM2-135M-Instruct", checkpoint=None):
    if os.path.exists(os.environ.get("DATA_PATH") + f"models/{model_name}_SFT_{checkpoint}_steps"):
        model = AutoModelForCausalLM.from_pretrained(
            os.environ.get("DATA_PATH") + f"models/{model_name}_SFT_{checkpoint}_steps",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(os.environ.get("DATA_PATH") + f"models/{model_name}")
        logger.info(f"Succesfully loaded {model_name} checkpoint {checkpoint}!")
        
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            os.environ.get("DATA_PATH") + f"models/{model_name}",
            torch_dtype="auto",
            device_map="auto"
        )
        checkpoint = get_last_checkpoint(os.environ.get("DATA_PATH") + f"output/{model_name}_SFT")
        model = PeftModel.from_pretrained(base_model, os.environ.get("DATA_PATH") + f"output/{model_name}_SFT/checkpoint-{checkpoint}")
        model = model.merge_and_unload()
        model.save_pretrained(os.environ.get("DATA_PATH") + f"models/{model_name}_SFT_{checkpoint}_steps")
        logger.info(f"Model saved to {os.environ.get('DATA_PATH')}models/{model_name}_SFT_{checkpoint}_steps")
        tokenizer = AutoTokenizer.from_pretrained(os.environ.get("DATA_PATH") + f"models/{model_name}")
        logger.info(f"Succesfully loaded {model_name} checkpoint {checkpoint}!")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.bos_token   
    return model, tokenizer

def load_dataset(task):
    df = pd.read_csv(os.environ.get("DATA_PATH") + f"input/{task}_training_data.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42) 
    train_df = train_df.sample(n=min(len(train_df), 100_000), random_state=42)
    val_df = val_df.sample(n=min(len(val_df), 4_000), random_state=42)

    train_data_dict = {"prompt": train_df["instruction"].tolist(), "completion": train_df["output"].tolist(), "ground_truth": train_df["output"].tolist()}
    eval_data_dict = {"prompt": val_df["instruction"].tolist(), "completion": val_df["output"].tolist(), "ground_truth": val_df["output"].tolist()}

    train_dataset = Dataset.from_dict(train_data_dict)
    eval_dataset = Dataset.from_dict(eval_data_dict)
    logger.info(f"Succesfully loaded train/eval datasets!")
    return train_dataset, eval_dataset
