import os
import logging
import yaml

from transformers import EarlyStoppingCallback
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from source.utils import load_dataset, load_model_checkpoint_and_tokenizer
from source.rewards import rewards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_grpo_trainer(model, train_dataset, eval_dataset, config):
    output_dir = os.environ.get("DATA_PATH") + f"output/{config['model_name']}_GRPO_{config['reward_fcts'][0]}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_args = GRPOConfig(output_dir=output_dir,
                               learning_rate=config["learning_rate"],
                               per_device_train_batch_size=config["per_device_train_batch_size"],
                               gradient_accumulation_steps=config["gradient_accumulation_steps"],
                               label_names=["labels"],
                               eval_strategy="steps",
                               num_generations=config["num_generations"],
                               eval_steps=config["eval_steps"],
                               save_steps=config["save_steps"],
                               save_total_limit=config["save_total_limit"],
                               logging_steps=config["logging_steps"],
                               load_best_model_at_end=True,
                               bf16=False,
                               fp16=True,
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

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        reward_funcs=[rewards[fct_name] for fct_name in config["reward_fcts"]],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["patience"])],
    )

    return trainer

if __name__ == "__main__":
    with open("configs/grpo_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model, tokenizer = load_model_checkpoint_and_tokenizer(config["model_name"], config["checkpoint"])
    train_dataset, eval_dataset = load_dataset("GRPO")
    trainer = get_grpo_trainer(model, train_dataset, eval_dataset, config)

    logger.info(f"Starting training...")
    trainer.train()
    logger.info(f"Training completed!")
