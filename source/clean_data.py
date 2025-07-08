import os
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_data(sft_weight=0.5):
    train_preflop_path = os.environ.get("DATA_PATH") + "input/preflop_60k_train_set_prompt_and_label.json"
    train_preflop_df = pd.read_json(train_preflop_path)

    train_postflop_path = os.environ.get("DATA_PATH") + "input/postflop_500k_train_set_prompt_and_label.json"
    train_postflop_df = pd.read_json(train_postflop_path)

    train_df = pd.concat([train_preflop_df, train_postflop_df], ignore_index=True)
    n_samples = len(train_df)
    logger.info(f"Total samples in the training set: {n_samples}")

    sft_samples = int(n_samples * sft_weight)
    sft_train_df = train_df.sample(n=sft_samples, random_state=42).reset_index(drop=True)
    grpo_train_df = train_df.drop(sft_train_df.index).reset_index(drop=True)

    sft_train_df.to_csv(os.environ.get("DATA_PATH") + "input/SFT_training_data.csv", index=False)
    grpo_train_df.to_csv(os.environ.get("DATA_PATH") + "input/GRPO_training_data.csv", index=False)

    logger.info(f"Training data cleaned and split into SFT and GRPO datasets ({len(sft_train_df)} SFT samples, {len(grpo_train_df)} GRPO samples).")

if __name__ == "__main__":
    clean_data()