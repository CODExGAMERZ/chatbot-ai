import json
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

INTENTS_FILE = os.path.join(DATA_DIR, "intents.json")

with open(INTENTS_FILE, "r", encoding="utf-8") as f:
    intents_data = json.load(f)

GENERIC_RESPONSE_TEMPLATES = [
    "Here’s what I know about that.",
    "Let me explain this for you.",
    "This is how it works.",
    "Here is a simple explanation.",
    "I can help explain this."
]

FOLLOW_UP_TEMPLATES = [
    "Let me know if you want more details.",
    "Feel free to ask a follow-up question.",
    "I can explain this further if you want."
]

def generate_responses(patterns, n=4):
    responses = []
    topic = patterns[0] if patterns else "this topic"

    for _ in range(n):
        base = random.choice(GENERIC_RESPONSE_TEMPLATES)
        follow = random.choice(FOLLOW_UP_TEMPLATES)
        responses.append(f"{base} {follow}")

    return list(set(responses))

added = 0

for intent in intents_data["intents"]:
    if "responses" in intent and len(intent["responses"]) == 0:
        intent["responses"] = generate_responses(intent.get("patterns", []))
        added += 1

with open(INTENTS_FILE, "w", encoding="utf-8") as f:
    json.dump(intents_data, f, indent=2)

print(f"Auto-generated responses for {added} intents.")

