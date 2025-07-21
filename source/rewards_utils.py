import re
from treys import Card, Evaluator

ACTION_REGEX = re.compile(r"^(fold|call|check|raise\s+\d+(\.\d+)?)$")

def is_valid_action(action):
    """Return True if action matches the accepted poker action format."""
    return bool(ACTION_REGEX.match(action.strip()))
    
name_map = {
    'ace': 'A',
    'king': 'K',
    'queen': 'Q',
    'jack': 'J',
    'ten': 'T',
    'nine': '9',
    'eight': '8',
    'seven': '7',
    'six': '6',
    'five': '5',
    'four': '4',
    'three': '3',
    'two': '2'
}

suit_map = {
    'diamond': 'd',
    'club': 'c',
    'spade': 's',
    'heart': 'h'
}

all_possible_cards = [f"{name} of {suit}".lower() for name in name_map for suit in suit_map]

def parse_state(prompt):
    hand_match = re.search(r"your holding is \[(.*?)\]", prompt, re.IGNORECASE)
    board_match = re.findall(r"The (flop|turn|river) comes (.*?)[\.\n]", prompt)

    hand = hand_match.group(1).split(" and ") if hand_match else []
    board = []
    for _, cards in board_match:
        possible_cards = [card.replace("and", "").replace("then", "").strip() for card in cards.split(",")]
        board += [c for c in possible_cards if c.lower() in all_possible_cards]

    return {"hand": hand, "board": board}

def strategy_match_score(pred, ref):
    if pred.strip().lower() == ref.strip().lower():
        return 1.0
    elif "fold" in pred and "fold" in ref:
        return 0.9
    elif "check" in pred and "check" in ref:
        return 0.9
    elif "bet" in pred and "bet" in ref:
        return 0.7
    else:
        return 0.0
    
def action_risk_score(action):
    action = action.lower()
    if "all-in" in action or "bet" in action:
        amount = int(re.search(r'\d+', action).group()) if re.search(r'\d+', action) else 5
        return min(1.0, amount / 100)  # scale to [0, 1]
    elif "check" in action:
        return 0.0
    elif "call" in action:
        return 0.2
    elif "fold" in action:
        return 0.1
    return 0.0

def card_to_treys(card_string):
    pattern = r"([a-zA-Z]+) of ([a-zA-Z]+)"
    match = re.match(pattern, card_string.strip(), re.IGNORECASE)
    if match:
        rank = name_map[match.group(1).lower()]
        suit = suit_map[match.group(2).lower()]
        return f"{rank}{suit}"
    return None

def parse_hand_and_board(input_str):
    m = re.search(r'holding is \[(.*?)\]', input_str, re.IGNORECASE)
    hole_cards = []
    if m:
        card_parts = m.group(1).split(' and ')
        for c in card_parts:
            hole_cards.append(card_to_treys(c))

    board = []
    flop = re.search(r'flop comes ([^,]+), ([^,]+), and ([^.]+)', input_str, re.IGNORECASE)
    if flop:
        for i in range(1, 4):
            board.append(card_to_treys(flop.group(i)))
    turn = re.search(r'turn comes ([^.]+)', input_str, re.IGNORECASE)
    if turn:
        board.append(card_to_treys(turn.group(1)))
    river = re.search(r'river comes ([^.]+)', input_str, re.IGNORECASE)
    if river:
        board.append(card_to_treys(river.group(1)))
    return hole_cards, board

def normalize_treys_score(score):
    """
    Normalize treys evaluator score (1 = best, 7462 = worst)
    to a hand strength (0.0 = worst, 1.0 = best)
    """
    return (7463 - score) / 7462

def evaluate_strength(hole_cards, board_cards):
    evaluator = Evaluator()
    hole = [Card.new(card_to_treys(card)) for card in hole_cards]
    board = [Card.new(card_to_treys(card)) for card in board_cards]
    score = evaluator.evaluate(board, hole)
    score = normalize_treys_score(score)
    return score