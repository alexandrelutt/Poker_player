# Poker Player

This project aims to fine-tune a small Language Model to play Poker using Supervised Fine Tuning as well as Group Relative Policy Optimization, as a follow-up to the paper [PokerBench: Training Large Language Models to become Professional Poker Players](https://arxiv.org/html/2501.08328v1). This repository is set up to run on a Google Cloud Virtual Machine.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

To get started with the Poker Player, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/alexandrelutt/Poker_player
   cd Poker_player
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Run setup script** (which handle packages installation and data downloading)

   ```bash
   . setup.sh
   ```

4. **If you want to make contributions, log in on GitHub (optional)** 
   ```bash
   sudo apt  install gh
   gh auth login
   ```

## Usage

For easier manipulation, it is suggested to run the following commands inside a tmux session, which can be done with

   ```bash
   tmux attach
   export DATA_PATH="/home/alexlutt/Poker_player/data/"
   ```   

1. **Run training pipeline** with arguments detailed in `configs/SFT_config.yaml` and `configs/GRPO_config.yaml`

   ```bash
   . train.sh
   ```   

2. **Run evaluation pipeline** with arguments detailed in `configs/evaluation_config.yaml`

   ```bash
   . eval.sh
   ```   