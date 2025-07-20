import re

# Shared regex and format filter
ACTION_REGEX = re.compile(r"^(fold|call|check|raise\s+\d+(\.\d+)?)$")

def is_valid_action(action):
    """Return True if action matches the accepted poker action format."""
    return bool(ACTION_REGEX.match(action.strip()))

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
    Rewards aggressive actions: raising (especially large raises), calling.
    Penalizes passive actions: folding, checking.
    Returns 0 reward for invalid formats.
    """
    rewards = []
    for action in completions:
        if not is_valid_action(action):
            rewards.append(0.0)
            continue

        action_type = action.split()[0]
        if action_type == 'raise':
            try:
                amount = float(action.split()[1])
                reward = 1.0 + 0.1 * amount
            except (IndexError, ValueError):
                reward = 1.0
            rewards.append(reward)
        elif action_type == 'call':
            rewards.append(0.5)
        elif action_type == 'check':
            rewards.append(-0.5)
        elif action_type == 'fold':
            rewards.append(-1.0)
        else:
            rewards.append(0.0)
    return rewards

rewards = {"risk_averse": risk_averse_reward,
           "risk_seeking": risk_seeking_reward}