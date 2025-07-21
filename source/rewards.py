from source.rewards_utils import is_valid_action, parse_state, strategy_match_score, evaluate_strength, action_risk_score

def risk_averse_reward(completions, ground_truth, **kwargs):
    """
    FOR TESTING GRPO TRAINING PIPELINE: successfully parses and rewards poker actions.
    Rewards safe, conservative actions: folding and checking.
    Penalizes aggressive actions: raising, calling.
    Returns 0 reward for invalid formats.
    """
    rewards = []
    for action in completions:
        if not is_valid_action(action):
            rewards.append(0.0)
            continue

        action_type = action.split()[0]
        if action_type == 'fold':
            rewards.append(1.0)
        elif action_type == 'check':
            rewards.append(0.5)
        elif action_type == 'call':
            rewards.append(-0.5)
        elif action_type == 'raise':
            rewards.append(-1.0)
        else:
            rewards.append(0.0)
    return rewards

def risk_seeking_reward(completions, ground_truth, **kwargs):
    """
    Reward function for Poker decision model using GRPO.
    Compares generated actions to ground truth, with strategic adjustment.

    Args:
        completions (list[str]): Generated actions from model (e.g., "bet 8").
        ground_truth (list[str]): Reference optimal actions.

    Returns:
        list[float]: Rewards, normalized to [-1, 1]
    """

    prompts = kwargs.get("prompts")  # list of scenario descriptions
    rewards = []

    for prompt, completion, truth in zip(prompts, completions, ground_truth):
        # 1. Direct match
        if completion.strip().lower() == truth.strip().lower():
            match_reward = 1.0
        else:
            match_reward = strategy_match_score(completion, truth)  # soft score in [0, 1]

        # 2. Strategic adjustment
        state = parse_state(prompt)
        strength = evaluate_strength(state["hand"], state["board"])
        risk = action_risk_score(completion)

        # Penalize reckless aggression (high risk with low strength)
        reckless_penalty = max(0.0, risk * (1.0 - strength))

        # Final reward
        total = match_reward - reckless_penalty
        total = max(-1.0, min(1.0, total))  # clip
        rewards.append(total)

    return rewards

rewards = {"risk_averse": risk_averse_reward,
           "risk_seeking": risk_seeking_reward}