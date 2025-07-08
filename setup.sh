export DATA_PATH="/home/easy_daily_articles/Poker_player/data/"

pip install -r requirements.txt

huggingface-cli download RZ412/PokerBench --repo-type dataset --local-dir data/input/
rm -rf data/input/.cache

mkdir data/models/Qwen2-0.5B-Instruct
huggingface-cli download Qwen/Qwen2-0.5B-Instruct --local-dir data/models/Qwen2-0.5B-Instruct
rm -rf data/models/Qwen2-0.5B-Instruct/.cache

mkdir data/models/SmolLM2-135M-Instruct
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct --local-dir data/models/SmolLM2-135M-Instruct
rm -rf data/models/SmolLM2-135M-Instruct/.cache

mkdir logs/
python3 -m source.clean_data > logs/data_cleaning.log 2>&1 &