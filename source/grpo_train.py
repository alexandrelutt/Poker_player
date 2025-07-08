import os
import logging
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel, LoraConfig
from source.data import load_dataset
from source.rewards import rewards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_checkpoint_and_tokenizer(model_name="SmolLM2-135M-Instruct"):
    if not os.path.exists(os.environ.get("DATA_PATH") + f"models/{model_name}_SFT"):
        base_model = AutoModelForCausalLM.from_pretrained(
        os.environ.get("DATA_PATH") + f"models/{model_name}",
        torch_dtype="auto",
        device_map="auto"
    	)
        checkpoint = get_last_checkpoint(os.environ.get("DATA_PATH") + f"models/{model_name}")
        model = PeftModel.from_pretrained(base_model, os.environ.get("DATA_PATH") + f"output/{model_name}/checkpoint-{checkpoint}")
        model = model.merge_and_unload()
        model.save_pretrained(os.environ.get("DATA_PATH") + f"models/{model_name}_SFT_{checkpoint}_steps")
        logger.info(f"Model saved to {os.environ.get('DATA_PATH')}models/{model_name}_SFT_{checkpoint}_steps")

    else:
        logger.info(f"Model already exists at {os.environ.get('DATA_PATH')}models/{model_name}_SFT_{checkpoint}_steps")
        model = AutoModelForCausalLM.from_pretrained(
            os.environ.get("DATA_PATH") + f"models/{model_name}_SFT",
            torch_dtype="auto",
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("DATA_PATH") + f"models/{model_name}")
    logger.info(f"Succesfully loaded {model_name} checkpoint {checkpoint}!")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.bos_token   
    return model, tokenizer

def get_grpo_trainer(model, train_dataset, eval_dataset, config):
    output_dir = os.environ.get("DATA_PATH") + f"output/{config['model_name']}_GRPO/"
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
    
    model, tokenizer = load_model_checkpoint_and_tokenizer(config["model_name"])
    train_dataset, eval_dataset = load_dataset("GRPO")
    trainer = get_grpo_trainer(model, train_dataset, eval_dataset, config)

    logger.info(f"Starting training...")
    trainer.train()
    logger.info(f"Training completed!")
