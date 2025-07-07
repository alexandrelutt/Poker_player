import os
import pandas as pd
import logging
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

logging.basicConfig(level=logging.INFO)
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

def load_dataset(config):
    train_data_path = os.environ.get("DATA_PATH") + "input/" + config["train_file"]

    df = pd.read_json(train_data_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_data_dict = {"prompt": train_df["instruction"].tolist(), "completion": train_df["output"].tolist()}
    eval_data_dict = {"prompt": val_df["instruction"].tolist(), "completion": val_df["output"].tolist()}

    train_dataset = Dataset.from_dict(train_data_dict)
    eval_dataset = Dataset.from_dict(eval_data_dict)
    logger.info(f"Succesfully loaded train/eval datasets!")
    return train_dataset, eval_dataset

def get_trainer(model, train_dataset, eval_dataset, config):
    output_dir = os.environ.get("DATA_PATH") + f"output/{config['model_name']}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_args = SFTConfig(output_dir=output_dir,
                               learning_rate=config["learning_rate"],
                               per_device_train_batch_size=config["per_device_train_batch_size"],
                               gradient_accumulation_steps=config["gradient_accumulation_steps"],
                               label_names=["labels"],
                               eval_strategy="steps",
                               eval_steps=config["eval_steps"],
                               save_steps=config["save_steps"],
                               save_total_limit=config["save_total_limit"],
                               logging_steps=config["logging_steps"],
                               load_best_model_at_end=True,
                               metric_for_best_model="eval_loss", 
                               report_to="tensorboard"
                            )
        
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_modules"],
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["patience"])],
    )

    return trainer

if __name__ == "__main__":
    with open("configs/SFT_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model, tokenizer = load_model_and_tokenizer(config["model_name"])
    train_dataset, eval_dataset = load_dataset(config)
    trainer = get_trainer(model, train_dataset, eval_dataset, config)

    logger.info(f"Starting training...")
    trainer.train()
    logger.info(f"Training completed!")
