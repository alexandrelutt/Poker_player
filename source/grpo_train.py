import os
import pandas as pd
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    with open("configs/GRPO_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    pass