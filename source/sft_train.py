import os
import logging
import yaml

from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from source.utils import load_dataset, load_model_and_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sft_trainer(model, train_dataset, eval_dataset, config):
    output_dir = os.environ.get("DATA_PATH") + f"output/{config['model_name']}_SFT/"
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
    with open("configs/sft_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model, tokenizer = load_model_and_tokenizer(config["model_name"])
    train_dataset, eval_dataset = load_dataset(task="SFT")
    trainer = get_sft_trainer(model, train_dataset, eval_dataset, config)

    logger.info(f"Starting training...")
    trainer.train()
    logger.info(f"Training completed!")
