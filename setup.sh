export DATA_PATH="/home/alexlutt/Poker_player/data/"

mkdir -p ${DATA_PATH}
pip install -r requirements.txt

huggingface-cli download RZ412/PokerBench --repo-type dataset --local-dir data/input/
rm -rf data/input/.cache

mkdir -p data/models/Qwen2-0.5B-Instruct
huggingface-cli download Qwen/Qwen2-0.5B-Instruct --local-dir data/models/Qwen2-0.5B-Instruct
rm -rf data/models/Qwen2-0.5B-Instruct/.cache

mkdir -p data/models/SmolLM2-135M-Instruct
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct --local-dir data/models/SmolLM2-135M-Instruct
rm -rf data/models/SmolLM2-135M-Instruct/.cache

mkdir -p data/output
mkdir -p logs/
python3 -m source.clean_data > logs/data_cleaning.log 2>&1 &
